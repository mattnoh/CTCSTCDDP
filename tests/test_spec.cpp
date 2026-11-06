// Correctness checks for IntegratorRK4 and IntervalConstraint. Plain exit code.
//   1. IntegratorRK4 Jx, Ju vs central differences: fixed step (1 and 4 substeps), free
//      final time, and with a post-step projection.
//   2. IntervalConstraint Jx, Ju vs central differences: every Form, same step grids.
//   3. Non-augmented == augmented: the Density row equals y(t_k+1) - y(t_k) - eps for
//      the accumulator y' = H integrated by the SAME IntegratorRK4 on [x; y].
//   4. Curvature: the Gauss-Newton vhp is symmetric and PSD for lambda >= 0;
//      Curvature::None gives zero.
//
// Model: point mass with quadratic drag, v' = u + g - CD |v| v, so the RK4
// sensitivities are genuinely state dependent.
#include "ctcs_stc/interval_constraint.hpp"

#include <aligator/core/vector-space.hpp>

#include <cmath>
#include <cstdio>
#include <string>

using ctcs_stc::Form;
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;
using Vec3 = Eigen::Vector3d;
using Space = aligator::context::VectorSpace;

namespace {
constexpr int NX = 6;
constexpr double CD = 0.3, DT = 0.3;
const Vec3 GRAVITY(0.0, 0.0, -9.81);
int failures = 0;

void check(bool ok, const std::string &what, double value) {
  std::printf("[%s] %-62s %.3e\n", ok ? "pass" : "FAIL", what.c_str(), value);
  if (!ok) ++failures;
}

/// x = [p; v], u = [a (3)] or [a (3); Th].  Extra state rows (the accumulator) have
/// zero rate here; AugmentedODE fills them.
struct DragODE : aligator::dynamics::ODEAbstractTpl<double> {
  using Base = aligator::dynamics::ODEAbstractTpl<double>;
  using Data = aligator::dynamics::ContinuousDynamicsDataTpl<double>;
  DragODE(const Space &space, int nu) : Base(space, nu) {}
  void forward(const ConstVectorRef &x, const ConstVectorRef &u, Data &d) const override {
    const Vec3 v = x.segment<3>(3);
    d.xdot_.setZero();
    d.xdot_.head<3>() = v;
    d.xdot_.segment<3>(3) = u.head<3>() + GRAVITY - CD * v.norm() * v;
  }
  void dForward(const ConstVectorRef &x, const ConstVectorRef &, Data &d) const override {
    const Vec3 v = x.segment<3>(3);
    const double n = v.norm();
    d.Jx_.setZero();
    d.Jx_.block<3, 3>(0, 3).setIdentity();
    d.Jx_.block<3, 3>(3, 3) = -CD * (n * Eigen::Matrix3d::Identity() + v * v.transpose() / n);
    d.Ju_.setZero();
    d.Ju_.block<3, 3>(3, 0).setIdentity();
  }
};

// Three implications covering a nontrivial trigger, a control-dependent consequence
// and an always-on path constraint.
ctcs_stc::Spec makeSpec(Form form, int nu) {
  ctcs_stc::Spec spec(form, 1e-3);
  const Vec3 o(0.4, -0.2, 1.0);
  auto zx = [](const V &) { return V::Zero(NX).eval(); };
  auto zu = [nu](const V &, const V &) { return V::Zero(nu).eval(); };
  auto low = [](const V &x) { return x(2) - 1.5; };
  auto low_dx = [](const V &) { V g = V::Zero(NX); g(2) = 1; return g; };
  spec.add("below_h_slow")
      .triggerWhenNegative(low, low_dx)
      .consequence([](const V &x, const V &) { return x.segment<3>(3).squaredNorm() - 0.25; },
                   [](const V &x, const V &) { V g = V::Zero(NX); g.segment<3>(3) = 2 * x.segment<3>(3); return g; }, zu);
  spec.add("below_h_thrust")
      .triggerWhenNegative(low, low_dx)
      .consequence([](const V &, const V &u) { return u.head<3>().norm() / 9.0 - 1.0; },
                   [](const V &, const V &) { return V::Zero(NX).eval(); },
                   [nu](const V &, const V &u) {
                     V g = V::Zero(nu);
                     g.head<3>() = u.head<3>() / (9.0 * u.head<3>().norm());
                     return g;
                   })
      .scale(2.0)
      .weight(3.0);
  spec.add("keep_out")
      .triggerWhenNegative([](const V &) { return -1.0; }, zx)
      .consequence([o](const V &x, const V &) { return 1.0 - (x.head<3>() - o).norm() / 0.8; },
                   [o](const V &x, const V &) {
                     const Vec3 d = x.head<3>() - o;
                     V g = V::Zero(NX);
                     g.head<3>() = -d / (0.8 * d.norm());
                     return g;
                   },
                   zu);
  return spec;
}

struct Grid {
  int substeps;
  bool free_time;
  bool projected;
  const char *name;
};

ctcs_stc::IntegratorRK4 makeIntegrator(const Space &space, const Grid &g) {
  const int nu = g.free_time ? 4 : 3;
  ctcs_stc::Projection proj = g.projected ? ctcs_stc::normalizeSegment<double>(3, 3) : ctcs_stc::Projection{};
  if (g.free_time) return ctcs_stc::IntegratorRK4(DragODE(space, nu), ctcs_stc::FreeTime{3}, g.substeps, proj);
  return ctcs_stc::IntegratorRK4(DragODE(space, nu), DT, g.substeps, proj);
}

// A point whose interval enters the trigger region, the keep-out ball and the thrust limit.
void point(bool free_time, V &x, V &u) {
  x.resize(NX);
  x << 0.1, 0.1, 1.7, 0.6, -0.5, -1.2;
  u.resize(free_time ? 4 : 3);
  u.head<3>() << 2.0, -1.0, 13.0;
  if (free_time) u(3) = 0.35;
}

/// max |FD - J| / max |J| over both Jacobians of a map f(x,u).
template <class F>
double fdError(F f, const V &x, const V &u, const M &Jx, const M &Ju) {
  const double h = 1e-6;
  double err = 0.0, scale = 1e-12;
  for (int j = 0; j < x.size(); ++j) {
    V xp = x, xm = x;
    xp(j) += h;
    xm(j) -= h;
    const V fd = (f(xp, u) - f(xm, u)) / (2 * h);
    err = std::max(err, (fd - Jx.col(j)).cwiseAbs().maxCoeff());
    scale = std::max(scale, Jx.col(j).cwiseAbs().maxCoeff());
  }
  for (int j = 0; j < u.size(); ++j) {
    V up = u, um = u;
    up(j) += h;
    um(j) -= h;
    const V fd = (f(x, up) - f(x, um)) / (2 * h);
    err = std::max(err, (fd - Ju.col(j)).cwiseAbs().maxCoeff());
    scale = std::max(scale, Ju.col(j).cwiseAbs().maxCoeff());
  }
  return err / scale;
}

/// [x; y] with y' = H(x, u): the augmented CTCS of Elango 2025, as an Aligator ODE.
struct AugmentedODE : aligator::dynamics::ODEAbstractTpl<double> {
  using Base = aligator::dynamics::ODEAbstractTpl<double>;
  using Data = aligator::dynamics::ContinuousDynamicsDataTpl<double>;
  DragODE base;
  ctcs_stc::Spec spec;
  AugmentedODE(const Space &aug, const Space &space, int nu, ctcs_stc::Spec s)
      : Base(aug, nu), base(space, nu), spec(std::move(s)) {}
  void forward(const ConstVectorRef &z, const ConstVectorRef &u, Data &d) const override {
    auto bd = base.createData();
    base.forward(z.head(NX), u, *bd);
    d.xdot_.head(NX) = bd->xdot_;
    d.xdot_(NX) = spec.density(V(z.head(NX)), V(u));
  }
  void dForward(const ConstVectorRef &, const ConstVectorRef &, Data &) const override {
    throw std::logic_error("value only");
  }
};
}  // namespace

int main() {
  const Space space(NX);
  V x, u;
  const Grid grids[] = {{1, false, false, "dt fixed, 1 substep"},
                        {4, false, false, "dt fixed, 4 substeps"},
                        {2, true, false, "free time, 2 substeps"},
                        {2, true, true, "free time, 2 substeps, projection"}};
  const struct { Form form; const char *name; } forms[] = {
      {Form::Original2018, "Original2018"}, {Form::Density, "Density"}, {Form::DensitySplit, "DensitySplit"}};

  // 1. Integrator Jacobians.
  for (const auto &g : grids) {
    const auto dyn = makeIntegrator(space, g);
    point(g.free_time, x, u);
    auto data = dyn.createData();
    dyn.dForward(x, u, *data);
    const M Jx = data->Jx(), Ju = data->Ju();
    auto step = [&](const V &xx, const V &uu) {
      dyn.forward(xx, uu, *data);
      return V(data->xnext_);
    };
    const double e = fdError(step, x, u, Jx, Ju);
    check(e < 1e-6, std::string("IntegratorRK4 FD Jacobians, ") + g.name, e);
  }

  // 2. Row Jacobians.
  for (const auto &f : forms)
    for (const auto &g : grids) {
      if (g.projected) continue;  // the row never sees the projection
      const auto dyn = makeIntegrator(space, g);
      const auto spec = makeSpec(f.form, dyn.nu);
      point(g.free_time, x, u);
      const ctcs_stc::IntervalConstraint fn(spec, dyn);
      auto data = fn.createData();
      fn.evaluate(x, u, *data);
      const bool active = data->value_.maxCoeff() > -spec.epsilon() + 1e-12 || f.form == Form::Original2018;
      fn.computeJacobians(x, u, *data);
      const M Jx = data->Jx_, Ju = data->Ju_;
      auto row = [&](const V &xx, const V &uu) {
        fn.evaluate(xx, uu, *data);
        return V(data->value_);
      };
      const double e = fdError(row, x, u, Jx, Ju);
      check(active && e < 1e-6, std::string("row FD Jacobians, ") + f.name + ", " + g.name, e);
    }

  // 3. Non-augmented row == augmented accumulator through the same integrator.
  for (const auto &g : grids) {
    if (g.projected) continue;
    const auto dyn = makeIntegrator(space, g);
    const auto spec = makeSpec(Form::Density, dyn.nu);
    point(g.free_time, x, u);
    const ctcs_stc::IntervalConstraint fn(spec, dyn);
    auto data = fn.createData();
    fn.evaluate(x, u, *data);
    const double I = data->value_(0) + spec.epsilon();

    const Space aug(NX + 1);
    AugmentedODE ode(aug, space, dyn.nu, spec);
    const ctcs_stc::IntegratorRK4 adyn = g.free_time
                                             ? ctcs_stc::IntegratorRK4(ode, ctcs_stc::FreeTime{3}, g.substeps)
                                             : ctcs_stc::IntegratorRK4(ode, DT, g.substeps);
    V z(NX + 1);
    z << x, 0.0;
    auto ad = adyn.createData();
    adyn.forward(z, u, *ad);
    const double y = ad->xnext_(NX);
    const double rel = std::abs(I - y) / std::max(std::abs(y), 1e-300);
    check(y > 0.0 && rel < 1e-12, std::string("Density row == RK4 accumulator, ") + g.name, rel);
  }

  // 4. Curvature.
  for (const auto &f : forms) {
    const Grid g{2, true, false, ""};
    const auto dyn = makeIntegrator(space, g);
    const auto spec = makeSpec(f.form, dyn.nu);
    point(true, x, u);
    const V lbda = V::Constant(spec.dim(), 0.7);
    const ctcs_stc::IntervalConstraint fn(spec, dyn);
    auto data = fn.createData();
    fn.computeJacobians(x, u, *data);
    fn.computeVectorHessianProducts(x, u, lbda, *data);
    const M H = data->vhp_buffer_;
    const double asym = (H - H.transpose()).cwiseAbs().maxCoeff();
    const double min_eig = Eigen::SelfAdjointEigenSolver<M>(0.5 * (H + H.transpose())).eigenvalues().minCoeff();
    const bool nonzero = f.form == Form::Original2018 || H.norm() > 0.0;
    check(asym <= 1e-12 * (1 + H.norm()) && min_eig >= -1e-10 * (1 + H.norm()) && nonzero,
          std::string("GN vhp symmetric PSD, ") + f.name, H.norm());

    const ctcs_stc::IntervalConstraint first_order(spec, dyn, ctcs_stc::Curvature::None);
    auto d2 = first_order.createData();
    first_order.computeVectorHessianProducts(x, u, lbda, *d2);
    check(d2->vhp_buffer_.isZero(0.0), std::string("Curvature::None gives zero vhp, ") + f.name,
          d2->vhp_buffer_.norm());
  }

  std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
