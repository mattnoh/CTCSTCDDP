The method is not new here: STC (Szmuk, Reynolds & Açıkmeşe, JGCD 2020), the integrated
exterior penalty with a per-interval ε and its pointwise bound (Elango et al., Automatica 2025,
eq. 34 and Thm. 10), and the squared density `[-g]_+^2 [c]_+^2` of CT-cSTC (Uzun, Açıkmeşe &
Carson, AIAA SciTech 2025; code at https://github.com/UW-ACL/CT-cSTC). The solver is
Aligator's ProxDDP (https://github.com/Simple-Robotics/aligator). BibTeX: `CITATIONS.bib`.

# Using ctcs_stc with Aligator

This folder holds the whole interface: three headers, no library to link besides Aligator.

| header | what it gives you |
|---|---|
| `spec.hpp` | `ctcs_stc::Spec`: the rules you write (triggers, consequences, scaling, form, ε) |
| `integrator_rk4.hpp` | `ctcs_stc::IntegratorRK4`: RK4 stage dynamics for an Aligator ODE (fixed or free step, substeps, projection) |
| `interval_constraint.hpp` | `ctcs_stc::IntervalConstraint`: the Spec as Aligator stage rows `c(x_k, u_k) <= 0` |

Include `ctcs_stc/interval_constraint.hpp`; it pulls in the other two.

The method is not new here: STC from Szmuk et al. 2018, the integrated exterior penalty with a
per-interval ε and its pointwise bound from Elango et al. 2025 (Automatica, eq. 34 and Thm. 10),
the squared density `[-g]_+^2 [c]_+^2` from Uzun et al. 2025.

---

## 1. Build and link

```bash
# Aligator (once), from the submodule (git submodule update --init --recursive)
cmake -S third_party/aligator -B third_party/aligator/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/install \
  -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCUMENTATION=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_WITH_PINOCCHIO_SUPPORT=OFF \
  -DCMAKE_PREFIX_PATH=<prefix with Eigen, Boost, fmt, mimalloc>     # e.g. a conda env
cmake --build third_party/aligator/build -j && cmake --install third_party/aligator/build

# your project
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/install;<deps prefix>"
export LD_LIBRARY_PATH=$PWD/install/lib:<deps prefix>/lib   # libaligator has RUNPATH=$ORIGIN
```
```cmake
find_package(aligator REQUIRED)
add_subdirectory(ctcs_stc)                      # or add include/ to your include path
target_link_libraries(my_ocp PRIVATE ctcs_stc::ctcs_stc)
```
If several Aligator installs are on the prefix path (a conda env often ships an older one),
put yours first and check `aligator_DIR` in `CMakeCache.txt`.

---

## 2. The model: write it once, as an Aligator ODE

```cpp
#include "ctcs_stc/interval_constraint.hpp"
using V = Eigen::VectorXd;

struct MyODE : aligator::dynamics::ODEAbstractTpl<double> {
  using Base = aligator::dynamics::ODEAbstractTpl<double>;
  using Data = aligator::dynamics::ContinuousDynamicsDataTpl<double>;
  explicit MyODE(const aligator::context::VectorSpace &space) : Base(space, NU) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u, Data &d) const override {
    d.xdot_ = ...;                         // f(x, u), size nx
  }
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u, Data &d) const override {
    d.Jx_ = ...;                           // df/dx, nx x nx
    d.Ju_ = ...;                           // df/du, nx x nu
  }
};
```
- **Vector-space states only** (`nx == ndx`). A quaternion is carried as a plain R^4 block and
  renormalized by a projection (section 3).
- **Analytic Jacobians.** They feed both the dynamics and the constraint's sensitivities, and
  the constraint is only as accurate as they are. Check them once against finite differences.
- Put **every control the model reads in `u`**, including a free step length (it is simply not
  read by `forward`; its column of `Ju` is zero).
- **Control scaling** (recommended when controls differ by orders of magnitude): let the solver
  see `ũ` with `u_phys = c + s .* ũ`, and apply it inside the ODE (`Ju = B * diag(s)`).
  Then the dynamics and the constraint see the same scaled control automatically
  (`examples/quadrotor.hpp`).

---

## 3. Stage dynamics: `IntegratorRK4`

```cpp
aligator::context::VectorSpace space(NX);

// fixed step DT, 4 RK4 substeps per interval
const ctcs_stc::IntegratorRK4 dyn(MyODE(space), DT, /*substeps=*/4);

// free final time: the interval length is u(j)
const ctcs_stc::IntegratorRK4 dyn(MyODE(space), ctcs_stc::FreeTime{j});

// with a projection after each full step (here: unit quaternion in x(6..9))
const ctcs_stc::IntegratorRK4 dyn(MyODE(space), ctcs_stc::FreeTime{j}, 1,
                                  ctcs_stc::normalizeSegment<double>(6, 4));
```
It is an ordinary `aligator::dynamics::ExplicitIntegratorAbstractTpl`, so it goes straight into
`aligator::context::StageModel(cost, dyn)`. Aligator itself offers Euler, semi-implicit Euler,
midpoint and RK2; RK4 is provided here because the constraint integrates with RK4 stage points.

| option | meaning |
|---|---|
| `DT` or `FreeTime{j}` | fixed step, or `h = u(j)`; the `d x_{k+1}/d h` column is added to `Ju` |
| `substeps` | M chained RK4 steps of `h/M` per interval |
| `Projection` | `{value, jacobian}` applied once after the full step; inner stage points are not projected |

With free time, **bound the step length yourself** (an ordinary stage constraint, e.g.
`(h - h_max)/(h_max - h_min) <= 0` and the lower bound), and carry `t` as a state with `ṫ = 1`
if you need time in the cost or the output.

---

## 4. The rules: `Spec`

```cpp
ctcs_stc::Spec spec(ctcs_stc::Form::Density, /*eps=*/1.0);

spec.add("landing_speed")                                    // g(x) < 0  ==>  c(x,u) <= 0
    .triggerWhenNegative(
        [](const V &x) { return x(2) - Z_TRIG; },              // g: fires below Z_TRIG
        [](const V &)  { V g = V::Zero(NX); g(2) = 1; return g; })
    .consequence(
        [](const V &x, const V &) { return x.segment<3>(3).squaredNorm() - VMAX * VMAX; },
        [](const V &x, const V &) { V g = V::Zero(NX); g.segment<3>(3) = 2 * x.segment<3>(3); return g; },
        [](const V &, const V &)  { return V::Zero(NU).eval(); })
    .scale(1.0 / (VMAX * VMAX))                                // consequence as a fraction of its limit
    .weight(1.0 / EPS);                                        // row reads  sum_i I_i / EPS - 1 <= 0
```

| builder call | effect |
|---|---|
| `add(name)` | start a new implication |
| `triggerWhenNegative(g, dg)` | trigger `g(x)`, active where negative; state only |
| `consequence(c, dcx, dcu)` | consequence `c(x,u) <= 0` with both gradients |
| `scale(s)` | use `s·c` everywhere; same zero set (put `1/limit` here) |
| `weight(w)` | weight in the density sum (`w [c]_+^2 == [√w c]_+^2`) |
| `epsilon(e)` | per-implication budget, `Form::DensitySplit` only |
| `finiteDifference()` | allow central differences for gradients this implication did not supply |

- **Plain path constraint (CTCS):** always-on trigger, `g = -1` with a zero gradient
  (`examples/pointmass_sphere.cpp`).
- **Gradients are required.** A missing one throws unless you opt in with
  `.finiteDifference()`. There is deliberately no autodiff (see the note in `spec.hpp`).
- Lambdas are **copied** with the spec into every stage: capture by value.

### Forms
| `Form` | rows per stage | what it enforces |
|---|---|---|
| `Density` | 1 | `∫ Σ_i w_i [-g_i]_+^2 [s_i c_i]_+^2 dt - ε <= 0` over the interval |
| `DensitySplit` | one per implication | `∫ w_i [-g_i]_+^2 [s_i c_i]_+^2 dt - ε_i <= 0` |
| `Original2018` | one per implication | nodal `[-g_i]_+ · s_i c_i <= 0`; nothing between nodes |

---

## 5. Add it to the stages and solve

```cpp
std::vector<xyz::polymorphic<aligator::context::StageModel>> stages;
for (int k = 0; k < N; ++k) {
  aligator::context::StageModel stage(cost, dyn);
  stage.addConstraint(Bounds(), aligator::NegativeOrthantTpl<double>());           // your ordinary rows
  stage.addConstraint(ctcs_stc::IntervalConstraint(spec, dyn),
                      aligator::NegativeOrthantTpl<double>());                     // rows are <= 0
  stages.emplace_back(stage);
}
aligator::context::TrajOptProblem problem(x0, stages, terminal_cost);

aligator::context::SolverProxDDP solver(/*tol=*/1e-4, /*mu_init=*/0.1, /*max_iters=*/3000,
                                        aligator::QUIET, aligator::StepAcceptanceStrategy::FILTER);
solver.setup(problem);
solver.run(problem, xs_guess, us_guess);
const auto &r = solver.results_;          // r.xs, r.us, r.num_iters, r.prim_infeas, r.dual_infeas
```
- **Always `NegativeOrthantTpl`**, never an equality set.
- `IntervalConstraint` **copies the integrator's ODE, step rule and substeps**, so the row is
  integrated along exactly the dynamics the solver steps. Pass the same `dyn` you give the
  `StageModel`. Because of that, the Density row equals RK4 on an accumulator `y' = H` over the
  step, without adding the state (`tests/test_spec.cpp` checks this to 1e-16).
- The spec is copied per stage; editing it afterwards does not reach the problem.

---

## 6. Scaling and ε (what decides whether it works)

1. **Each consequence as a fraction of its own limit** (`scale(1/limit)`), weights 1
   (Elango 2025, Sec. 6). One ε then means the same for every rule in a Density row:
   `|p - o| >= r  ->  1 - |p - o|/r`,  `|v| <= v_max  ->  |v|^2/v_max^2 - 1`.
   For a length residual (a glideslope in metres), divide by the length you care about.
2. **Write the row as `Σ I / ε - 1 <= 0`** (spec ε = 1, weights `1/ε`), so the row is O(1)
   for the augmented Lagrangian. In the quadrotor example a raw row of size ~1e-6 was
   effectively feasible to ProxDDP from the start and the rules were not enforced.
3. **Choose ε from the bound you need.** Elango 2025, Thm. 10 bounds the pointwise violation in
   terms of ε; `pointmass_sphere` uses it to set the margin δ added to the consequence.
4. **Scale the problem, not only the row:** costs O(1) (the quadrotor needed ×1e-2), controls
   in similar units (section 2). In the quadrotor example these mattered as much as ε.

---

## 7. Curvature

`IntervalConstraint` provides the Gauss-Newton term of the rows in
`computeVectorHessianProducts` (full symmetric block). ProxDDP calls it only with
`HessianApprox::EXACT`; under the default `GAUSS_NEWTON` the rows are first order.
`IntervalConstraint(spec, dyn, ctcs_stc::Curvature::None)` forces first order under EXACT too.

---

## 8. Auditing a solution

The solver's own convergence flag is not a compliance check. After a solve:
```cpp
// x: any state (a node r.xs[k], or a point of the resampled flow), u = r.us[k]
double H  = spec.density(x, u);                         // all rules
double Hi = spec.density(x, u, i);                      // rule i only
bool   on = spec.triggered(i, x);                       // g_i(x) < 0
const auto &im = spec.implication(i);
double ci = spec.consequenceVal(im, x, u);              // s_i c_i(x,u): <= 0 means satisfied
```
Resample each interval with the same model (several fine RK4 steps) and check every triggered
consequence there, not only at the nodes. `examples/stc_landing.cpp` writes node and dense CSVs
for an external grader.

---

## 9. Limits and pitfalls

- Vector-space states only; no manifold `⊕`.
- Controls are **zero-order hold** inside an interval.
- `LD_LIBRARY_PATH` must contain Aligator's lib and its dependencies' lib at run time.
- Results can be **sensitive to tuning and rounding**: in the quadrotor example, changes that
  only moved the last bits of a Jacobian reshuffled which initial conditions failed. Judge a
  setting on many initial conditions and keep a held-out set; 10 is not enough.
- ProxDDP often stops at the iteration cap with a feasible trajectory on hard STC problems;
  check compliance (section 8) rather than the flag.

---

## 10. Examples in this repo

| file | shows |
|---|---|
| `examples/pointmass_sphere.cpp` + `point_mass.hpp` | CTCS path constraint, fixed step, 4 substeps, ε from Thm. 10 |
| `examples/stc_landing.cpp` + `quadrotor.hpp` | 8 STC rules in one Density row, free final time, quaternion projection, control and cost scaling, CSV export |
| `tests/test_spec.cpp` | how to check your own model: integrator and row Jacobians vs FD, row == augmented accumulator |
