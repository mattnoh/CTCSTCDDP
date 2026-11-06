// State-triggered landing of a Crazyflie-class quadrotor, free final time, ProxDDP.
//
// Port of the single-horizon CT-cSTC landing OCP (stationary pad at the origin) to
// Aligator: same model, rules, bounds and horizon. All the STC rules sit in ONE
// ctcs_stc::Spec (Form::Density), so each interval carries a single CTCS row.
//
//   always on      tilt <= 11 deg,  |omega| <= 0.35 rad/s
//   staging        |r_xy| > 1.1 m  ==>  z >= 1.8 m        (hold altitude until over the pad)
//   line of sight  z < 1.8 m       ==>  pad inside a 30 deg body-z cone
//   landing        z < 0.8 m       ==>  |v| <= 0.75 m/s, tilt <= 0.95 deg, |omega| <= 0.019 rad/s,
//                                       |r_xy| <= tan(15 deg) (z + 0.1)   (glideslope)
//
// Scaling for ProxDDP (the ALIPDDP profile's weights do not transfer):
//   * each consequence is divided by its own limit (Elango 2025, Sec. 6), weight 1, so
//     the row is  sum_i I_i / EPS_INT - 1 <= 0  with every I_i a dimensionless fraction;
//   * controls in scaled units (quadrotor.hpp), cost multiplied by COST_SCALE;
//   * light step-length regularization R_theta (theta - 0.10)^2, tuned for this example.
//
// Usage: stc_landing [r0x r0y r0z v0x v0y v0z [csv_prefix]]     (default 2.4 2.0 2.0 0 -2 0)
//   csv_prefix writes <prefix>.csv (nodes) and <prefix>_dense.csv (20 samples per interval)
//   in the column layout of the thesis graders.
#include "quadrotor.hpp"

#include "ctcs_stc/interval_constraint.hpp"

#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/constraints/negative-orthant.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace quadrotor;

namespace {
constexpr int N = 80;
constexpr double TH0 = 0.07, TH_MIN = 0.05, TH_MAX = 0.12;  // step length [s]

constexpr double FMIN = 0.12, FMAX = 0.52, HOVER = MASS * 9.81;
const double TAU_MAX = 0.046 * (0.60 / 4.0 - 0.08 / 4.0) * J_SCALE;
constexpr double V_MAX = 3.0;  // general speed cap

// Costs (physical units), then scaled as a whole.
constexpr double W_TIME = 0.04, R_THR = 2.0, R_MOM = 0.6, R_THETA = 3.0, TH_REF = 0.10;
constexpr double TW_POS = 2500.0, TW_VEL = 900.0, TW_TILT = 200.0, TW_OM = 50.0;
constexpr double COST_SCALE = 1e-2;

// STC rules
const double DEG = M_PI / 180.0;
const double TILT_PATH = 11.0 * DEG, OMEGA_PATH = 0.35;
constexpr double STAGE_RADIUS = 1.1, STAGE_SCALE = 0.25, Z_STAGE = 1.8;
constexpr double LOS_ALT = 1.8, LOS_SCALE = 0.25;
const double LOS_TAN = std::tan(30.0 * DEG);
constexpr double ALT_TRIG = 0.8, V_LAND = 0.75, OMEGA_LAND = 0.019, GS_APEX = 0.1;
const double TILT_LAND = 0.95 * DEG, GS_TAN = std::tan(15.0 * DEG);

// CTCS: per-interval budget on the sum of the dimensionless integrals, and the length
// the glideslope residual [m] is measured in.
constexpr double EPS_INT = 1e-6, GS_LENGTH = 0.1;

// Solver
constexpr double TOL = 1e-4, MU_INIT = 0.1;
constexpr std::size_t MAX_ITERS = 3000;

V physicalControl(const V &u) { V p = u; p.head(NUP) = physical(u); return p; }

double qperp2(const V &x) { return x(7) * x(7) + x(8) * x(8); }  // sin^2(tilt / 2)
double tiltOf(const V &x) { return 2.0 * std::asin(std::min(1.0, std::sqrt(qperp2(x)))); }
double rxy(const V &x) { return std::sqrt(x(0) * x(0) + x(1) * x(1) + 1e-12); }
Vec3 bodyZ(const V &x) {
  const double w = x(6), a = x(7), b = x(8), c = x(9);
  return Vec3(2 * (a * c + w * b), 2 * (b * c - w * a), w * w - a * a - b * b + c * c);
}
// Lateral distance of the pad from the body-z axis minus tan(30 deg) times the along-axis
// distance: <= 0 means the pad is inside the line-of-sight cone.
double losResidual(const V &x) {
  const Vec3 bz = bodyZ(x), r = x.head<3>();
  const double ax = r.dot(bz);
  return std::sqrt(std::max(0.0, r.squaredNorm() - ax * ax) + 1e-12) - LOS_TAN * ax;
}

// ── Costs ────────────────────────────────────────────────────────────────────
/// COST_SCALE (W_TIME theta + R_theta (theta - 0.1)^2 + R_THR (f - mg)^2 + R_MOM |M|^2).
struct StageCost : aligator::CostAbstractTpl<double> {
  using Base = aligator::CostAbstractTpl<double>;
  using CostData = aligator::CostDataAbstractTpl<double>;
  explicit StageCost(const aligator::context::VectorSpace &s) : Base(s, NU) {}
  void evaluate(const ConstVectorRef &, const ConstVectorRef &uu, CostData &d) const override {
    const V u = physicalControl(uu);
    d.value_ = COST_SCALE * (W_TIME * u(IDX_THETA) + R_THETA * std::pow(u(IDX_THETA) - TH_REF, 2) +
                             R_THR * std::pow(u(0) - HOVER, 2) + R_MOM * u.segment<3>(1).squaredNorm());
  }
  void computeGradients(const ConstVectorRef &, const ConstVectorRef &uu, CostData &d) const override {
    const V u = physicalControl(uu);
    d.Lx_.setZero();
    d.Lu_.setZero();
    d.Lu_(0) = 2 * R_THR * (u(0) - HOVER);
    d.Lu_.segment<3>(1) = 2 * R_MOM * u.segment<3>(1);
    d.Lu_.head(NUP) = d.Lu_.head(NUP).cwiseProduct(U_SCALE());
    d.Lu_(IDX_THETA) = W_TIME + 2 * R_THETA * (u(IDX_THETA) - TH_REF);
    d.Lu_ *= COST_SCALE;
  }
  void computeHessians(const ConstVectorRef &, const ConstVectorRef &, CostData &d) const override {
    d.Lxx_.setZero();
    d.Lxu_.setZero();
    d.Luu_.setZero();
    d.Luu_(0, 0) = 2 * R_THR * U_SCALE()(0) * U_SCALE()(0);
    for (int j = 1; j < NUP; ++j) d.Luu_(j, j) = 2 * R_MOM * U_SCALE()(j) * U_SCALE()(j);
    d.Luu_(IDX_THETA, IDX_THETA) = 2 * R_THETA;
    d.Luu_ *= COST_SCALE;
  }
};

/// COST_SCALE 0.5 (wp |r|^2 + wv |v|^2 + wt (qx^2 + qy^2) + wo |omega|^2): land at rest, level.
struct TerminalCost : aligator::CostAbstractTpl<double> {
  using Base = aligator::CostAbstractTpl<double>;
  using CostData = aligator::CostDataAbstractTpl<double>;
  V w;
  explicit TerminalCost(const aligator::context::VectorSpace &s) : Base(s, 0), w(V::Zero(NX)) {
    w.segment<3>(0).setConstant(TW_POS);
    w.segment<3>(3).setConstant(TW_VEL);
    w(7) = w(8) = TW_TILT;
    w.segment<3>(10).setConstant(TW_OM);
    w *= COST_SCALE;
  }
  void evaluate(const ConstVectorRef &x, const ConstVectorRef &, CostData &d) const override {
    d.value_ = 0.5 * (w.array() * x.array().square()).sum();
  }
  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &, CostData &d) const override {
    d.Lx_ = w.cwiseProduct(x);
    d.Lu_.setZero();
  }
  void computeHessians(const ConstVectorRef &, const ConstVectorRef &, CostData &d) const override {
    d.Lxx_ = w.asDiagonal();
    d.Lxu_.setZero();
    d.Luu_.setZero();
  }
};

// ── Ordinary bounds (rows <= 0, each scaled to an O(1) margin) ───────────────
struct Bounds : aligator::StageFunctionTpl<double> {
  using Base = aligator::StageFunctionTpl<double>;
  using Data = aligator::StageFunctionDataTpl<double>;
  static constexpr int NR = 7;
  Bounds() : Base(NX, NU, NR) {}
  void evaluate(const ConstVectorRef &x, const ConstVectorRef &uu, Data &d) const override {
    const V u = physicalControl(uu);
    d.value_ << (FMIN - u(0)) / (FMAX - FMIN), (u(0) - FMAX) / (FMAX - FMIN),
        u.segment<3>(1).squaredNorm() / (TAU_MAX * TAU_MAX) - 1.0,
        (u(IDX_THETA) - TH_MAX) / (TH_MAX - TH_MIN), (TH_MIN - u(IDX_THETA)) / (TH_MAX - TH_MIN),
        -x(2), x.segment<3>(3).squaredNorm() / (V_MAX * V_MAX) - 1.0;
  }
  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &uu, Data &d) const override {
    const V u = physicalControl(uu);
    d.Jx_.setZero();
    d.Ju_.setZero();
    d.Ju_(0, 0) = -1.0 / (FMAX - FMIN);
    d.Ju_(1, 0) = 1.0 / (FMAX - FMIN);
    d.Ju_.block<1, 3>(2, 1) = 2.0 * u.segment<3>(1).transpose() / (TAU_MAX * TAU_MAX);
    d.Ju_(3, IDX_THETA) = 1.0 / (TH_MAX - TH_MIN);
    d.Ju_(4, IDX_THETA) = -1.0 / (TH_MAX - TH_MIN);
    d.Jx_(5, 2) = -1.0;
    d.Jx_.block<1, 3>(6, 3) = 2.0 * x.segment<3>(3).transpose() / (V_MAX * V_MAX);
    for (int j = 0; j < NUP; ++j) d.Ju_.col(j) *= U_SCALE()(j);  // chain rule through the scaling
  }
};

// ── The STC / CTCS row ───────────────────────────────────────────────────────
ctcs_stc::Spec landingRules() {
  // eps = 1 with weights 1/EPS_INT: the row reads sum_i I_i / EPS_INT - 1 <= 0.
  ctcs_stc::Spec spec(ctcs_stc::Form::Density, 1.0);
  const double w = 1.0 / EPS_INT;
  const double lt = std::pow(std::sin(0.5 * TILT_PATH), 2), ld = std::pow(std::sin(0.5 * TILT_LAND), 2);
  auto Z = [] { return V::Zero(NX).eval(); };
  auto never_u = [](const V &, const V &) { return V::Zero(NU).eval(); };
  auto always = [](const V &) { return -1.0; };
  auto always_dx = [Z](const V &) { return Z(); };
  auto land = [](const V &x) { return x(2) - ALT_TRIG; };
  auto land_dx = [Z](const V &) { V g = Z(); g(2) = 1.0; return g; };
  auto tilt_dx = [Z](const V &x, const V &) { V g = Z(); g(7) = 2 * x(7); g(8) = 2 * x(8); return g; };
  auto om_dx = [Z](const V &x, const V &) { V g = Z(); g.segment<3>(10) = 2 * x.segment<3>(10); return g; };

  spec.add("path_tilt").triggerWhenNegative(always, always_dx)
      .consequence([lt](const V &x, const V &) { return qperp2(x) - lt; }, tilt_dx, never_u)
      .scale(1.0 / lt).weight(w);
  spec.add("path_omega").triggerWhenNegative(always, always_dx)
      .consequence([](const V &x, const V &) { return x.segment<3>(10).squaredNorm() - OMEGA_PATH * OMEGA_PATH; },
                   om_dx, never_u)
      .scale(1.0 / (OMEGA_PATH * OMEGA_PATH)).weight(w);
  spec.add("staging")
      .triggerWhenNegative([](const V &x) { return -(rxy(x) - STAGE_RADIUS) / STAGE_SCALE; },
                           [Z](const V &x) {
                             V g = Z();
                             g(0) = -x(0) / (rxy(x) * STAGE_SCALE);
                             g(1) = -x(1) / (rxy(x) * STAGE_SCALE);
                             return g;
                           })
      .consequence([](const V &x, const V &) { return Z_STAGE - x(2); },
                   [Z](const V &, const V &) { V g = Z(); g(2) = -1.0; return g; }, never_u)
      .scale(1.0 / STAGE_SCALE).weight(w);
  spec.add("line_of_sight")
      .triggerWhenNegative([](const V &x) { return (x(2) - LOS_ALT) / LOS_SCALE; },
                           [Z](const V &) { V g = Z(); g(2) = 1.0 / LOS_SCALE; return g; })
      .consequence([](const V &x, const V &) { return losResidual(x); },
                   [Z](const V &x, const V &) {
                     V g = Z();
                     const double qw = x(6), qx = x(7), qy = x(8), qz = x(9);
                     const Vec3 bz = bodyZ(x), r = x.head<3>();
                     const double ax = r.dot(bz), raw = r.squaredNorm() - ax * ax;
                     const double lat = std::sqrt(std::max(0.0, raw) + 1e-12);
                     const double d_n2 = raw > 0 ? 0.5 / lat : 0.0;
                     const double d_ax = (raw > 0 ? -ax / lat : 0.0) - LOS_TAN;
                     g.head<3>() = 2 * d_n2 * r + d_ax * bz;
                     g(6) = d_ax * (r(0) * 2 * qy - r(1) * 2 * qx + r(2) * 2 * qw);
                     g(7) = d_ax * (r(0) * 2 * qz - r(1) * 2 * qw - r(2) * 2 * qx);
                     g(8) = d_ax * (r(0) * 2 * qw + r(1) * 2 * qz - r(2) * 2 * qy);
                     g(9) = d_ax * (r(0) * 2 * qx + r(1) * 2 * qy + r(2) * 2 * qz);
                     return g;
                   },
                   never_u)
      .scale(1.0 / LOS_SCALE).weight(w);
  spec.add("landing_speed").triggerWhenNegative(land, land_dx)
      .consequence([](const V &x, const V &) { return x.segment<3>(3).squaredNorm() - V_LAND * V_LAND; },
                   [Z](const V &x, const V &) { V g = Z(); g.segment<3>(3) = 2 * x.segment<3>(3); return g; }, never_u)
      .scale(1.0 / (V_LAND * V_LAND)).weight(w);
  spec.add("landing_tilt").triggerWhenNegative(land, land_dx)
      .consequence([ld](const V &x, const V &) { return qperp2(x) - ld; }, tilt_dx, never_u)
      .scale(1.0 / ld).weight(w);
  spec.add("landing_omega").triggerWhenNegative(land, land_dx)
      .consequence([](const V &x, const V &) { return x.segment<3>(10).squaredNorm() - OMEGA_LAND * OMEGA_LAND; },
                   om_dx, never_u)
      .scale(1.0 / (OMEGA_LAND * OMEGA_LAND)).weight(w);
  spec.add("landing_glideslope").triggerWhenNegative(land, land_dx)
      .consequence([](const V &x, const V &) { return rxy(x) - GS_TAN * (x(2) + GS_APEX); },
                   [Z](const V &x, const V &) {
                     V g = Z();
                     g(0) = x(0) / rxy(x);
                     g(1) = x(1) / rxy(x);
                     g(2) = -GS_TAN;
                     return g;
                   },
                   never_u)
      .scale(1.0 / GS_LENGTH).weight(w);

  return spec;
}

// ── Audit (plain maxima; the thesis graders read the CSV export) ─────────────
struct Audit {
  double tilt_path = 0, om_path = 0, speed = 0;
  double stage_short = 0, los = -1e9;
  double land_speed = 0, land_tilt = 0, land_om = 0, gs = -1e9;
  void add(const V &x) {
    tilt_path = std::max(tilt_path, tiltOf(x));
    om_path = std::max(om_path, x.segment<3>(10).norm());
    speed = std::max(speed, x.segment<3>(3).norm());
    if (rxy(x) > STAGE_RADIUS) stage_short = std::max(stage_short, Z_STAGE - x(2));
    if (x(2) < LOS_ALT) los = std::max(los, losResidual(x));
    if (x(2) < ALT_TRIG) {
      land_speed = std::max(land_speed, x.segment<3>(3).norm());
      land_tilt = std::max(land_tilt, tiltOf(x));
      land_om = std::max(land_om, x.segment<3>(10).norm());
      gs = std::max(gs, rxy(x) - GS_TAN * (x(2) + GS_APEX));
    }
  }
  void print(const char *name) const {
    std::printf("  %-6s path: tilt %5.2f/%.0f deg  omega %.3f/%.2f  speed %.2f/%.0f | staging shortfall %.4f m | "
                "LOS %+.4f m\n         landing: speed %.3f/%.2f  tilt %.3f/%.2f deg  omega %.4f/%.3f  glideslope %+.4f m\n",
                name, tilt_path / DEG, TILT_PATH / DEG, om_path, OMEGA_PATH, speed, V_MAX, stage_short, los,
                land_speed, V_LAND, land_tilt / DEG, TILT_LAND / DEG, land_om, OMEGA_LAND, gs);
  }
};

/// One RK4 step of the physical model with zero-order-hold control, quaternion renormalized.
V rk4(const V &x, const V &up, double h) {
  V xp = x.head(NXP);
  const V k1 = xdot(xp, up), k2 = xdot(xp + 0.5 * h * k1, up), k3 = xdot(xp + 0.5 * h * k2, up),
          k4 = xdot(xp + h * k3, up);
  xp += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
  xp.segment<4>(6).normalize();
  V xn = x;
  xn.head(NXP) = xp;
  return xn;
}

/// Thesis-grader CSVs: nodes, and SUB sub-samples per interval.
void writeCsv(const std::string &prefix, const ctcs_stc::Spec &rules, const std::vector<V> &xs,
              const std::vector<V> &us) {
  const char *hdr = "node,t_acc,rx,ry,rz,rvx,rvy,rvz,qw,qx,qy,qz,wx,wy,wz,fz_B,Mx,My,Mz,Theta,ctcs_integrand\n";
  std::FILE *fn = std::fopen((prefix + ".csv").c_str(), "w");
  std::FILE *fd = std::fopen((prefix + "_dense.csv").c_str(), "w");
  if (!fn || !fd) { std::fprintf(stderr, "cannot write %s*.csv\n", prefix.c_str()); std::exit(2); }
  std::fputs(hdr, fn);
  std::fputs(hdr, fd);
  auto row = [&](std::FILE *f, int node, double t, const V &x, const V &u, const V &up) {
    std::fprintf(f, "%d,%.10f", node, t);
    for (int i = 0; i < NXP; ++i) std::fprintf(f, ",%.10f", x(i));
    for (int i = 0; i < NUP; ++i) std::fprintf(f, ",%.10f", up(i));
    std::fprintf(f, ",%.10f,%.10e\n", up(IDX_THETA), rules.density(x, u));
  };
  constexpr int SUB = 20;
  double t = 0;
  for (int k = 0; k <= N; ++k) {
    const V &u = us[std::min(k, N - 1)];
    const V up = physicalControl(u);
    row(fn, k, t, xs[k], u, up);
    if (k == N) { row(fd, k, t, xs[k], u, up); break; }
    V x = xs[k];
    const double h = up(IDX_THETA) / SUB;
    for (int i = 0; i < SUB; ++i) {
      row(fd, k, t + i * h, x, u, up);
      x = rk4(x, up.head(NUP), h);
    }
    t += up(IDX_THETA);
  }
  std::fclose(fn);
  std::fclose(fd);
}
}  // namespace

int main(int argc, char **argv) {
  double ic[6] = {2.4, 2.0, 2.0, 0.0, -2.0, 0.0};
  if (argc >= 7)
    for (int i = 0; i < 6; ++i) ic[i] = std::atof(argv[i + 1]);
  const std::string csv_prefix = argc >= 8 ? argv[7] : "";
  V x0 = V::Zero(NX);
  x0 << ic[0], ic[1], ic[2], ic[3], ic[4], ic[5], 1, 0, 0, 0, 0, 0, 0, 0;

  aligator::context::VectorSpace space(NX);
  // Free time: the interval length is u(IDX_THETA); quaternion renormalized after each step.
  const ctcs_stc::IntegratorRK4 dynamics(ODE(space), ctcs_stc::FreeTime{IDX_THETA}, 1, quaternionProjection());
  StageCost cost(space);
  TerminalCost terminal(space);
  const auto rules = landingRules();

  std::vector<xyz::polymorphic<aligator::context::StageModel>> stages;
  for (int k = 0; k < N; ++k) {
    aligator::context::StageModel stage(cost, dynamics);
    stage.addConstraint(Bounds(), aligator::NegativeOrthantTpl<double>());
    stage.addConstraint(ctcs_stc::IntervalConstraint(rules, dynamics), aligator::NegativeOrthantTpl<double>());
    stages.emplace_back(stage);
  }
  aligator::context::TrajOptProblem problem(x0, stages, terminal);

  // Seed: hover thrust (f~ = 0), step TH0, and the rollout it produces (floored at z = 0).
  V u0 = V::Zero(NU);
  u0(IDX_THETA) = TH0;
  std::vector<V> us(N, u0), xs(N + 1, x0);
  auto data = dynamics.createData();
  auto &dd = static_cast<aligator::ExplicitDynamicsDataTpl<double> &>(*data);
  for (int k = 0; k < N; ++k) {
    dynamics.forward(xs[k], us[k], dd);
    xs[k + 1] = dd.xnext_;
    xs[k + 1](2) = std::max(xs[k + 1](2), 0.0);
  }

  aligator::context::SolverProxDDP solver(TOL, MU_INIT, MAX_ITERS);
  solver.setNumThreads(1);
  solver.setup(problem);
  const auto t0 = std::chrono::steady_clock::now();
  const bool converged = solver.run(problem, xs, us);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const auto &r = solver.results_;

  if (!csv_prefix.empty()) writeCsv(csv_prefix, rules, r.xs, r.us);

  Audit nodes, dense;
  double T = 0;
  for (int k = 0; k < N; ++k) {
    const V up = physicalControl(r.us[k]);
    nodes.add(r.xs[k]);
    dense.add(r.xs[k]);
    V x = r.xs[k];
    for (int i = 0; i < 40; ++i) dense.add(x = rk4(x, up.head(NUP), up(IDX_THETA) / 40));
    T += up(IDX_THETA);
  }
  nodes.add(r.xs[N]);
  std::printf("IC r0=(%.2f %.2f %.2f) v0=(%.2f %.2f %.2f)\n", ic[0], ic[1], ic[2], ic[3], ic[4], ic[5]);
  std::printf("solver flag=%d iters=%zu time=%.0f ms prim=%.1e dual=%.1e  T*=%.3f s  terminal |r|=%.2e |v|=%.2e\n",
              converged, static_cast<std::size_t>(r.num_iters), ms, r.prim_infeas, r.dual_infeas, T,
              r.xs[N].head<3>().norm(), r.xs[N].segment<3>(3).norm());
  nodes.print("nodes");
  dense.print("dense");
  return 0;
}
