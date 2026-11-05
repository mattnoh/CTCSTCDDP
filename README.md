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

along the continuous dynamics starting at `x_k`, and emits `I_k - eps <= 0`. `I_k` depends on
`(x_k, u_k)` only, so **no accumulator state is added**. This is the non-augmented form
(Elango 2025, eq. 34). It also holds for nonlinear dynamics, and it is numerically identical to
integrating an extra state `y' = H` with the same RK4 steps (checked in `tests/`).

## Layout
```
include/ctcs_stc/spec.hpp            the constraint (Eigen only): implications, forms, quadrature, derivatives
include/ctcs_stc/stage_function.hpp  Aligator adapter (aligator::StageFunctionTpl)
examples/pointmass_sphere.cpp        CTCS: point mass past a sphere (Elango 2025 setup)
examples/stc_landing.cpp             STC: glideslope + landing speed below a trigger altitude
examples/point_mass.hpp              dynamics/costs shared by the examples and tests
tests/test_spec.cpp                  FD Jacobians, non-augmented == RK4 accumulator, adapter == spec
third_party/aligator                 Aligator as a git submodule (upstream main)
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
./build/stc_landing
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

### 1. Write the constraint
```cpp
#include "ctcs_stc/stage_function.hpp"
using V = Eigen::VectorXd;

ctcs_stc::Spec<double> spec(ctcs_stc::Form::Density, /*eps=*/1.0);

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

### 2. Give it the continuous dynamics
```cpp
ctcs_stc::IntervalDynamics<double> dyn;
dyn.xdot = [](const V& x, const V& u, double tau) { ... };   // f(x,u)
dyn.fx   = [](const V& x, const V& u, double tau) { ... };   // df/dx
dyn.fu   = [](const V& x, const V& u, double tau) { ... };   // df/du
dyn.dt = DT;            // fixed step ...
// dyn.idx_theta = j;   // ... or free time: u(j) is the interval length
dyn.substeps = 2;       // RK4 steps per interval used by the quadrature
spec.setIntervalDynamics(dyn);
```
**This must be the same model your discrete dynamics steps.** The integral is taken along
`xdot` from `x_k`. If that differs from the solver's `x_{k+1} = f_d(x_k, u_k)`, the
constraint is checked along a trajectory the solver isn't actually following.

### 3. Add it to the stages and solve
```cpp
for (int k = 0; k < N; ++k) {
  aligator::context::StageModel stage(cost, dynamics);
  stage.addConstraint(ctcs_stc::StageFunction(nx, nu, spec),
                      aligator::NegativeOrthantTpl<double>());     // rows are <= 0
  stages.emplace_back(stage);
}
aligator::context::TrajOptProblem problem(x0, stages, terminal_cost);
aligator::context::SolverProxDDP solver(1e-5, 1e-2, 500);
solver.setup(problem);
solver.run(problem, xs_guess, us_guess);
```
Each stage gets its own copy of the spec. Editing `spec` after this point doesn't reach the
stages.

## Rules that decide whether it works
- **Scale every consequence to its own limit** (`1 - |p-o|/r`, `|a|/a_max - 1`, or
  `.scale(1/limit)`), so one `eps` means the same thing for every implication in a row.
- **Choose `eps` from the bound you need.** For `[c]_+^2` the pointwise violation is bounded
  as in Elango 2025, Theorem 10. `pointmass_sphere` uses it to set the margin `delta` that keeps
  the exact flow outside the sphere.
- **Controls that vary inside an interval** (first-order hold) need `IntervalDynamics::control`
  (+ `control_dx`, `control_du`), with the left knot carried in the state. Zero-order hold
  needs nothing.
- **Curvature.** `StageFunction` provides the Gauss-Newton term of the row through
  `computeVectorHessianProducts`. ProxDDP only calls it with `HessianApprox::EXACT`. Under the
  default `GAUSS_NEWTON` the rows are first-order. `Curvature::None` forces first-order.

## Results of the shipped examples
| example | result |
|---|---|
| `pointmass_sphere` | converges, 36 iterations; densely resampled clearance 13.07 mm (margin delta 11.70 mm) |
| `stc_landing` | glideslope satisfied between nodes; **does not converge at tol 1e-6** (see below) |

## Limitations
- **Vanishing gradient at the trigger surface.** The density `[-g]_+^2 [c]_+^2` and its gradient
  are zero exactly where the trigger switches on. If the optimum wants the consequence active
  right at that surface (landing speed at the trigger altitude, for instance), the active row
  has a near-zero gradient. You then see a residual violation at trigger onset (about 0.14 m/s
  in `stc_landing`) and ProxDDP stalls on dual feasibility. It does converge at tol 1e-4.
  This comes from the multiplicative STC form, not from the adapter.
- Vector-space states only (`ndx == nx`). Quaternion states would need the tangent-space
  Jacobian.
- The spec caches its last evaluation on exact `(x, u)`. If the interval dynamics read
  externals that change between solves, rebuild the stages.

## Author and license
Author: `<AUTHOR>`. There is no license file yet. Without one, others cannot legally reuse the
code. Add one (Aligator uses BSD-2) before publishing if you want it reused.
