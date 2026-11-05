// CTCS path constraint: fixed-time point mass flying past a sphere (the obstacle
// setup of Elango et al. 2025), keep-out enforced between nodes by one interval
// row per stage, solved with ProxDDP.
//
// Path constraint as an always-on implication (g = -1), consequence written as a
// fraction of the radius with Elango's Theorem 10 margin delta:
//     c = (R - |p - o| + delta) / R <= 0
// Row: I_k / EPS - 1 <= 0  (weight 1/EPS, spec eps 1).
#include "point_mass.hpp"

#include "ctcs_stc/stage_function.hpp"

#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/constraints/negative-orthant.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace point_mass;

namespace {
constexpr int K = 10, SUBSTEPS = 4;
constexpr double TF = 4.0, DT = TF / K, RADIUS = 2.0, EPS = 1e-8, OMEGA = 10.0;
const Vec3 OBSTACLE(2.5 + 0.5 / std::sqrt(2.0), 2.5 - 0.5 / std::sqrt(2.0), 5.5);
// Elango 2025 Theorem 10: pointwise violation <= (4 eps omega / R)^(1/3) R for this density.
const double DELTA = RADIUS * std::cbrt(4.0 * EPS * OMEGA / RADIUS);

ctcs_stc::Spec<double> keepOut() {
  ctcs_stc::Spec<double> spec(ctcs_stc::Form::Density, 1.0);
  spec.add("sphere")
      .triggerWhenNegative([](const V &) { return -1.0; }, [](const V &) { return V::Zero(NX).eval(); })
      .consequence(
          [](const V &x, const V &) { return (RADIUS - (x.head<3>() - OBSTACLE).norm() + DELTA) / RADIUS; },
          [](const V &x, const V &) {
            const Vec3 d = x.head<3>() - OBSTACLE;
            V g = V::Zero(NX);
            g.head<3>() = -d / (RADIUS * d.norm());
            return g;
          },
          [](const V &, const V &) { return V::Zero(NU).eval(); })
      .weight(1.0 / EPS);
  spec.setIntervalDynamics(intervalDynamics(DT, SUBSTEPS));
  return spec;
}
}  // namespace

int main() {
  V x0(NX), xf(NX);
  x0 << 5, 5, 10, -0.2, -0.2, -0.3;
  xf << 0, 0, 1, 0, 0, 0;
  Space space(NX);
  Dynamics dynamics(space, DT);
  EffortCost effort(space, 1e-3 * DT, V::Zero(NU));
  TerminalCost terminal(space, 1e3, xf);
  const auto spec = keepOut();

  std::vector<xyz::polymorphic<aligator::context::StageModel>> stages;
  for (int k = 0; k < K; ++k) {
    aligator::context::StageModel stage(effort, dynamics);
    stage.addConstraint(ctcs_stc::StageFunction(NX, NU, spec), aligator::NegativeOrthantTpl<double>());
    stages.emplace_back(stage);
  }
  aligator::context::TrajOptProblem problem(x0, stages, terminal);

  std::vector<V> xs(K + 1), us(K, V(-GRAVITY));
  for (int k = 0; k <= K; ++k) xs[k] = x0 + (xf - x0) * double(k) / K;
  aligator::context::SolverProxDDP solver(1e-5, 1e-2, 500);
  solver.rollout_type_ = aligator::RolloutType::LINEAR;
  solver.setNumThreads(1);
  solver.setup(problem);
  const bool converged = solver.run(problem, xs, us);
  const auto &r = solver.results_;

  // Independent check: clearance on 1000 samples of the exact flow per interval.
  double clearance = 1e9;
  for (int k = 0; k < K; ++k)
    for (int i = 0; i <= 1000; ++i)
      clearance = std::min(clearance, (flow(r.xs[k], r.us[k], DT * i / 1000.0).head<3>() - OBSTACLE).norm() - RADIUS);

  std::printf("converged=%d iters=%zu prim=%.1e dual=%.1e terminal_err=%.2e\n", converged,
              static_cast<std::size_t>(r.num_iters), r.prim_infeas, r.dual_infeas,
              (r.xs.back() - xf).head<3>().norm());
  std::printf("dense clearance = %.2f mm  (margin delta = %.2f mm; >= 0 means no violation)\n",
              1e3 * clearance, 1e3 * DELTA);
  return converged && clearance >= 0.0 ? 0 : 1;
}
