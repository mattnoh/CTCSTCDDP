// State-triggered constraints: a point mass lands on a pad at the origin. Once it is
// below h_gs it must stay inside a glideslope cone and below the landing speed:
//
//     z < h_gs  ==>  |p_xy| <= tan(theta) (z + apex)      (glideslope)
//     z < h_gs  ==>  |v|    <= v_land                     (landing speed)
//
// Both implications go into ONE Density row per interval (one spec), with
// dimensionless consequences, and are enforced between nodes. The OCP is solved
// twice, without and with the row, and the triggered violation is measured on a
// dense resampling of the exact flow.
#include "point_mass.hpp"

#include "ctcs_stc/stage_function.hpp"

#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/constraints/negative-orthant.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace point_mass;

namespace {
constexpr int K = 40, SUBSTEPS = 2;
constexpr double TF = 4.0, DT = TF / K;
constexpr double H_GS = 1.5;   // trigger altitude [m]
constexpr double L_TRIG = 0.25; // trigger scale [m]
const double TAN_GS = std::tan(30.0 * M_PI / 180.0);
constexpr double APEX = 0.1;    // cone apex below the pad [m]
constexpr double L_GS = 0.25;   // glideslope scale [m]
constexpr double V_LAND = 1.0;  // [m/s]
constexpr double EPS = 1e-5;    // interval budget

ctcs_stc::Spec<double> landingRules() {
  ctcs_stc::Spec<double> spec(ctcs_stc::Form::Density, 1.0);  // row = sum I / EPS - 1
  auto trig = [](const V &x) { return (x(2) - H_GS) / L_TRIG; };
  auto dtrig = [](const V &) { V g = V::Zero(NX); g(2) = 1.0 / L_TRIG; return g; };
  auto zero_u = [](const V &, const V &) { return V::Zero(NU).eval(); };

  spec.add("glideslope")
      .triggerWhenNegative(trig, dtrig)
      .consequence(
          [](const V &x, const V &) {
            return (std::sqrt(x.head<2>().squaredNorm() + 1e-12) - TAN_GS * (x(2) + APEX)) / L_GS;
          },
          [](const V &x, const V &) {
            V g = V::Zero(NX);
            g.head<2>() = x.head<2>() / std::sqrt(x.head<2>().squaredNorm() + 1e-12) / L_GS;
            g(2) = -TAN_GS / L_GS;
            return g;
          },
          zero_u)
      .weight(1.0 / EPS);

  spec.add("landing_speed")
      .triggerWhenNegative(trig, dtrig)
      .consequence([](const V &x, const V &) { return x.tail<3>().squaredNorm() / (V_LAND * V_LAND) - 1.0; },
                   [](const V &x, const V &) {
                     V g = V::Zero(NX);
                     g.tail<3>() = 2.0 * x.tail<3>() / (V_LAND * V_LAND);
                     return g;
                   },
                   zero_u)
      .weight(1.0 / EPS);

  spec.setIntervalDynamics(intervalDynamics(DT, SUBSTEPS));
  return spec;
}

struct Violation {
  double glideslope = 0.0, speed = 0.0;  // worst triggered excess [m], [m/s]
};

Violation denseViolation(const std::vector<V> &xs, const std::vector<V> &us) {
  Violation w;
  for (std::size_t k = 0; k < us.size(); ++k)
    for (int i = 0; i <= 200; ++i) {
      const V y = flow(xs[k], us[k], DT * i / 200.0);
      if (y(2) >= H_GS) continue;
      w.glideslope = std::max(w.glideslope, y.head<2>().norm() - TAN_GS * (y(2) + APEX));
      w.speed = std::max(w.speed, y.tail<3>().norm() - V_LAND);
    }
  return w;
}

bool solve(bool with_stc, Violation &out) {
  V x0(NX), xf = V::Zero(NX);
  x0 << 3, 2, 4, 0, 0, 0;
  Space space(NX);
  Dynamics dynamics(space, DT);
  EffortCost effort(space, 0.1 * DT, V(-GRAVITY));
  TerminalCost terminal(space, 1e3, xf);
  const auto spec = landingRules();

  std::vector<xyz::polymorphic<aligator::context::StageModel>> stages;
  for (int k = 0; k < K; ++k) {
    aligator::context::StageModel stage(effort, dynamics);
    if (with_stc)
      stage.addConstraint(ctcs_stc::StageFunction(NX, NU, spec), aligator::NegativeOrthantTpl<double>());
    stages.emplace_back(stage);
  }
  aligator::context::TrajOptProblem problem(x0, stages, terminal);

  std::vector<V> xs(K + 1), us(K, V(-GRAVITY));
  for (int k = 0; k <= K; ++k) xs[k] = x0 + (xf - x0) * double(k) / K;
  aligator::context::SolverProxDDP solver(1e-6, 1e-2, 500);
  solver.rollout_type_ = aligator::RolloutType::LINEAR;
  solver.setNumThreads(1);
  solver.setup(problem);
  const bool converged = solver.run(problem, xs, us);
  const auto &r = solver.results_;
  out = denseViolation(r.xs, r.us);
  std::printf("%-9s converged=%d iters=%3zu terminal_err=%.2e  triggered excess: glideslope=%7.2f mm  "
              "speed=%7.2f mm/s\n",
              with_stc ? "with STC" : "no STC", converged, static_cast<std::size_t>(r.num_iters),
              (r.xs.back() - xf).norm(), 1e3 * std::max(out.glideslope, 0.0), 1e3 * std::max(out.speed, 0.0));
  return converged;
}
}  // namespace

int main() {
  Violation free_run, stc_run;
  const bool a = solve(false, free_run);
  const bool b = solve(true, stc_run);
  return a && b ? 0 : 1;
}
