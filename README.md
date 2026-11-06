# ctcs_stc — continuous-time and state-triggered constraints for Aligator

A header-only C++ interface that adds two kinds of path constraint to
[Aligator](https://github.com/Simple-Robotics/aligator)'s constrained solver, ProxDDP:

- **CTCS**: continuous-time constraint satisfaction. A path constraint
  `c(x,u) <= 0` is enforced *between* the discretization nodes, not only at them.
- **STC**: state-triggered constraints, `g(x) < 0  ==>  c(x,u) <= 0`
  ("below 1.5 m, stay inside the glideslope cone").

Each one becomes an ordinary Aligator stage constraint, a row `c_k(x_k, u_k) <= 0` on stage
`k`, so the solver itself is unchanged.

## The method (credit where it is due)
None of the formulation is new here. This repo only implements it for DDP.

- STC and its nodal form `[-g]_+ * c <= 0`: Szmuk, Reynolds & Açıkmeşe, 2018.
- Continuous-time enforcement by an integrated exterior penalty with an `eps` budget per
  interval, and the pointwise violation bound (Theorem 10): Elango et al., *Automatica* 2025.
- Squared-positive-part STC density `[-g]_+^2 [c]_+^2`, integrated per interval: Uzun,
  Açıkmeşe & Carson, AIAA SciTech 2025 (CT-cSTC).

For interval `k` the interface computes

```
I_k = ∫_{t_k}^{t_k+1} H(x(t), u(t)) dt,     H = Σ_i w_i [-g_i(x)]_+^2 [s_i c_i(x,u)]_+^2
```

along the stage's own dynamics starting at `x_k` (the RK4 stage points of
`ctcs_stc::IntegratorRK4`, Simpson weights), and emits `I_k - eps <= 0`. `I_k` depends on
`(x_k, u_k)` only, so **no accumulator state is added**. This is the non-augmented form
(Elango 2025, eq. 34). It also holds for nonlinear dynamics, and it is numerically identical to
integrating an extra state `y' = H` with the same RK4 steps (checked in `tests/`).

## Layout
```
include/ctcs_stc/spec.hpp                 authoring: implications, forms, eps, pointwise density
include/ctcs_stc/integrator_rk4.hpp       RK4 integrator for Aligator ODE models (free time, substeps, projection)
include/ctcs_stc/interval_constraint.hpp  the rows: an aligator::StageFunctionTpl built from a Spec and the integrator
include/ctcs_stc/USAGE.md                 how to use the interface with Aligator
examples/pointmass_sphere.cpp             CTCS: point mass past a sphere (Elango 2025 setup)
examples/stc_landing.cpp                  STC: quadrotor landing, free final time, 8 rules in one CTCS row
examples/point_mass.hpp                   point-mass ODE and costs (pointmass_sphere)
examples/quadrotor.hpp                    quadrotor ODE, control scaling, quaternion projection (stc_landing)
tests/test_spec.cpp                       FD Jacobians (integrator, rows), row == augmented accumulator, curvature
third_party/aligator                      Aligator as a git submodule (upstream main)
```

## Building

### 1. Aligator
Pick one of the two.

**From the submodule** (C++ only, no Python):
```bash
git clone --recursive <this repo> && cd <this repo>      # or: git submodule update --init --recursive
cmake -S third_party/aligator -B third_party/aligator/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/install \
  -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCUMENTATION=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_WITH_PINOCCHIO_SUPPORT=OFF \
  -DCMAKE_PREFIX_PATH=<prefix with Eigen, Boost, fmt, mimalloc>     # e.g. a conda env
cmake --build third_party/aligator/build -j && cmake --install third_party/aligator/build
```
Aligator's dependencies are listed in its README. A conda env with
`eigen boost fmt mimalloc` covers them.

**Prebuilt:** `conda install -c conda-forge aligator`. conda-forge can lag behind upstream main,
so check the version.

### 2. This repo
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/install;<deps prefix>"
cmake --build build -j
export LD_LIBRARY_PATH=$PWD/install/lib:<deps prefix>/lib
ctest --test-dir build --output-on-failure
./build/pointmass_sphere
./build/stc_landing                                   # default IC
./build/stc_landing r0x r0y r0z v0x v0y v0z [csv_prefix]   # own IC, optional CSV export
```
`LD_LIBRARY_PATH` is needed because `libaligator.so` carries `RUNPATH=$ORIGIN`. An
executable's rpath therefore cannot resolve libaligator's own dependencies (libfmt, libmimalloc).

If more than one Aligator is on `CMAKE_PREFIX_PATH` (for example the conda env ships an older
one), put the install you want first and check `aligator_DIR` in `build/CMakeCache.txt`.

### Use it from your own project
```cmake
add_subdirectory(ctcs_stc)                  # or copy include/ into your tree
target_link_libraries(my_ocp PRIVATE ctcs_stc::ctcs_stc)
```
C++17. There is no Python binding. The triggers and consequences are C++ lambdas called inside
the quadrature, so a Python callback there would be slow.

## Writing an OCP

Full usage guide (API tables, options, scaling, auditing, pitfalls): [`include/ctcs_stc/USAGE.md`](include/ctcs_stc/USAGE.md).

### 1. Write the model once, as an Aligator ODE
```cpp
#include "ctcs_stc/interval_constraint.hpp"
using V = Eigen::VectorXd;

struct MyODE : aligator::dynamics::ODEAbstractTpl<double> {
  using Base = aligator::dynamics::ODEAbstractTpl<double>;
  using Data = aligator::dynamics::ContinuousDynamicsDataTpl<double>;
  explicit MyODE(const aligator::context::VectorSpace& space) : Base(space, NU) {}
  void forward(const ConstVectorRef& x, const ConstVectorRef& u, Data& d) const override { d.xdot_ = ...; }
  void dForward(const ConstVectorRef& x, const ConstVectorRef& u, Data& d) const override { d.Jx_ = ...; d.Ju_ = ...; }
};
```

### 2. Make the stage dynamics from it
```cpp
aligator::context::VectorSpace space(NX);
const ctcs_stc::IntegratorRK4 dynamics(MyODE(space), DT, /*substeps=*/4);          // fixed step
// const ctcs_stc::IntegratorRK4 dynamics(MyODE(space), ctcs_stc::FreeTime{j});    // free time: h = u(j)
// optional 4th argument: a projection after each step, e.g. ctcs_stc::normalizeSegment<double>(6, 4)
```
`IntegratorRK4` is an ordinary `aligator::dynamics::ExplicitIntegratorAbstractTpl`. Aligator
itself ships Euler, semi-implicit Euler, midpoint and RK2; RK4 is added here because the row's
quadrature is the RK4 one.

### 3. Write the constraint
```cpp
ctcs_stc::Spec spec(ctcs_stc::Form::Density, /*eps=*/1.0);

spec.add("landing_speed")                                     // one implication
    .triggerWhenNegative(
        [](const V& x) { return (x(2) - 1.5) / 0.25; },          // g(x): fires when z < 1.5 m
        [](const V&)   { V g = V::Zero(6); g(2) = 1 / 0.25; return g; })
    .consequence(
        [](const V& x, const V&) { return x.tail<3>().squaredNorm() / (VMAX*VMAX) - 1; },   // c(x,u) <= 0
        [](const V& x, const V&) { V g = V::Zero(6); g.tail<3>() = 2*x.tail<3>()/(VMAX*VMAX); return g; },
        [](const V&, const V&)   { return V::Zero(3).eval(); })
    .weight(1.0 / EPS);                                       // row = Σ I / EPS - 1  (dimensionless)
```
- **Plain path constraint (CTCS):** use an always-on trigger, `g = -1` with zero gradient
  (`examples/pointmass_sphere.cpp`).
- **Several implications** in one `Spec` share one row (`Form::Density`). Use
  `Form::DensitySplit` for one row per implication, each with its own `.epsilon(e)`.
  `Form::Original2018` is the nodal form, with no inter-sample enforcement.
- **Jacobians are required.** Pass analytic `dg/dx`, `dc/dx` and `dc/du`, or call
  `.finiteDifference()` on that implication to opt in to central differences. A missing
  Jacobian throws. Autodiff is deliberately not offered; see the note in `spec.hpp`.

### 4. Add it to the stages and solve
```cpp
for (int k = 0; k < N; ++k) {
  aligator::context::StageModel stage(cost, dynamics);
  stage.addConstraint(ctcs_stc::IntervalConstraint(spec, dynamics),
                      aligator::NegativeOrthantTpl<double>());     // rows are <= 0
  stages.emplace_back(stage);
}
aligator::context::TrajOptProblem problem(x0, stages, terminal_cost);
aligator::context::SolverProxDDP solver(1e-5, 1e-2, 500);
solver.setup(problem);
solver.run(problem, xs_guess, us_guess);
```
`IntervalConstraint` copies the integrator's ODE, step rule and substeps, so the row is always
integrated along the dynamics the solver steps; you cannot hand it a different model. Each
stage gets its own copy of the spec; editing `spec` afterwards doesn't reach the stages.

## Rules that decide whether it works
- **Scale every consequence to its own limit** (`1 - |p-o|/r`, `|a|/a_max - 1`, or
  `.scale(1/limit)`), so one `eps` means the same thing for every implication in a row.
- **Choose `eps` from the bound you need.** For `[c]_+^2` the pointwise violation is bounded
  as in Elango 2025, Theorem 10. `pointmass_sphere` uses it to set the margin `delta` that keeps
  the exact flow outside the sphere.
- **Controls are held constant over the interval** (zero-order hold). A first-order-hold
  hook existed in the ALIPDDP port and is not carried over.
- **Curvature.** `IntervalConstraint` provides the Gauss-Newton term of the row through
  `computeVectorHessianProducts`. ProxDDP only calls it with `HessianApprox::EXACT`. Under the
  default `GAUSS_NEWTON` the rows are first-order. `Curvature::None` forces first-order.

## Results of the shipped examples
| example | result |
|---|---|
| `pointmass_sphere` | converges, 36 iterations; densely resampled clearance 13.07 mm (margin delta 11.70 mm) |
| `stc_landing` | 100 sampled ICs (thesis benchmark samples, thesis node audit): 90/100 land and satisfy every rule (tuned on s00-s29: 27/30; held out s30-s99: 63/70). Default IC lands, compliant. ProxDDP's own convergence flag is rarely set (11/100); most runs stop at the 3000-iteration cap with a feasible, landed trajectory. |

`stc_landing` reproduces the single-horizon landing OCP of the CT-cSTC thesis code. The weights
of that profile do not transfer to ProxDDP; what changed is scaling only: each consequence
divided by its own limit with weight 1 (Elango 2025, Sec. 6), the interval budget `EPS_INT`,
the step-length regularization of the thesis defaults, cost x1e-2, and controls in scaled
units. The rules themselves are unchanged.

Tuning (scaling and solver options only): `EPS_INT = 1e-6` and Aligator's filter step
acceptance (`StepAcceptanceStrategy::FILTER`). The solve is sensitive to rounding and to
which ICs a setting is tuned on: changes that only move the last bits of the dynamics
Jacobians reshuffle which ICs fail, so judge a setting on many ICs, not a handful.

## Limitations
- Vector-space states only (`ndx == nx`). `stc_landing` carries the quaternion as a plain
  R^4 block renormalized after each step, not as a manifold state.
- The spec caches its last evaluation on exact `(x, u)`. If the interval dynamics read
  externals that change between solves, rebuild the stages.

## Author and license
Author: `<AUTHOR>`. There is no license file yet. Without one, others cannot legally reuse the
code. Add one (Aligator uses BSD-2) before publishing if you want it reused.
