# ctcs_stc — continuous-time and state-triggered constraints for Aligator

A header-only C++ interface that adds two kinds of path constraint to
[Aligator](https://github.com/Simple-Robotics/aligator)'s constrained solver, ProxDDP:

- **CTCS**: continuous-time constraint satisfaction. A path constraint
  `c(x,u) <= 0` is enforced *between* the discretization nodes, not only at them.
- **STC**: state-triggered constraints, `g(x) < 0  ==>  c(x,u) <= 0`
  ("below 1.5 m, stay inside the glideslope cone").

Each one becomes an ordinary Aligator stage constraint, a row `c_k(x_k, u_k) <= 0` on stage
`k`, so the solver itself is unchanged.

![Quadrotor landing: 3D trajectory and state histories with limits and triggers](docs/stc_landing.png)

`examples/stc_landing.cpp`: a quadrotor landing with eight state-triggered rules in one CTCS row
per interval, free final time, solved by ProxDDP.

## Method
This repo implements published methods; it does not introduce a new formulation:
state-triggered constraints (STC, Szmuk et al.), continuous-time constraint satisfaction
(CTCS, Elango et al.), and continuous-time compound state-triggered constraints (CT-cSTC,
Uzun et al., [UW-ACL/CT-cSTC](https://github.com/UW-ACL/CT-cSTC)). The solver is
[Aligator](https://github.com/Simple-Robotics/aligator)'s ProxDDP, used unmodified as a git
submodule. See [Citing](#citing).

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
CITATIONS.bib                             BibTeX: Aligator/ProxDDP, STC, CTCS, CT-cSTC
examples/pointmass_sphere.cpp             CTCS: point mass past a sphere (Elango 2025 setup)
examples/stc_landing.cpp                  STC: quadrotor landing, free final time, 8 rules in one CTCS row
examples/point_mass.hpp                   point-mass ODE and costs (pointmass_sphere)
examples/quadrotor.hpp                    quadrotor ODE, control scaling, quaternion projection (stc_landing)
tests/test_spec.cpp                       FD Jacobians (integrator, rows), row == augmented accumulator, curvature
aligator/                                 Aligator as a git submodule (upstream main)
docs/                                     figures
```

## Building

### 1. Aligator
Pick one of the two.

**From the submodule** (C++ only, no Python):
```bash
git clone --recursive git@github.com:mattnoh/CTCSTCDDP.git && cd CTCSTCDDP   # or: git submodule update --init --recursive
cmake -S aligator -B aligator/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/install \
  -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCUMENTATION=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_WITH_PINOCCHIO_SUPPORT=OFF \
  -DCMAKE_PREFIX_PATH=<prefix with Eigen, Boost, fmt, mimalloc>     # e.g. a conda env
cmake --build aligator/build -j && cmake --install aligator/build
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

## Usage
See [`include/ctcs_stc/USAGE.md`](include/ctcs_stc/USAGE.md) for how to write an OCP with this interface.

## Results of the shipped examples
| example | result |
|---|---|
| `pointmass_sphere` | converges, 36 iterations; densely resampled clearance 13.07 mm (margin delta 11.70 mm) |
| `stc_landing` | default start r0 = (2.4, 2.0, 2.0), v0 = (0, -2, 0): converges, 1110 iterations, T* = 8.5 s, lands within 0.1 mm. Densely resampled, the path, staging, line-of-sight, glideslope and landing angular-rate rules hold; landing speed and tilt exceed their limits by at most 0.005 m/s and 0.1 deg. Other initial conditions may need retuning. |

`stc_landing` reproduces the single-horizon landing OCP of the CT-cSTC thesis code. The weights
of that profile do not transfer to ProxDDP; what changed is scaling only: each consequence
divided by its own limit with weight 1 (Elango 2025, Sec. 6), the interval budget `EPS_INT`,
a light step-length regularization, cost x1e-2, and controls in scaled
units. The rules themselves are unchanged.

Settings: `EPS_INT = 1e-6`, `R_THETA = 3`, Aligator's default step acceptance.

## Limitations
- Vector-space states only (`ndx == nx`). `stc_landing` carries the quaternion as a plain
  R^4 block renormalized after each step, not as a manifold state.
- The spec caches its last evaluation on exact `(x, u)`. If the interval dynamics read
  externals that change between solves, rebuild the stages.

## Citing
If you use this interface, please cite the works it builds on:

```bibtex
% Solver
@misc{aligatorweb,
  author = {Jallet, Wilson and Bambade, Antoine and El Kazdadi, Sarah and Carpentier, Justin and Mansard, Nicolas},
  title  = {aligator},
  url    = {https://github.com/Simple-Robotics/aligator}
}

@article{jalletPROXDDPProximalConstrained2025,
  title   = {PROXDDP: Proximal Constrained Trajectory Optimization},
  author  = {Jallet, Wilson and Bambade, Antoine and Arlaud, Etienne and {El-Kazdadi}, Sarah and Mansard, Nicolas and Carpentier, Justin},
  year    = {2025},
  journal = {IEEE Transactions on Robotics},
  volume  = {41},
  pages   = {2605--2624},
  doi     = {10.1109/TRO.2025.3554437}
}

% State-triggered constraints (STC)
@article{szmukSuccessiveConvexificationRealTime2020,
  title   = {Successive Convexification for Real-Time 6-{DoF} Powered Descent Guidance with State-Triggered Constraints},
  author  = {Szmuk, Michael and Reynolds, Taylor P. and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  journal = {Journal of Guidance, Control, and Dynamics},
  volume  = {43},
  number  = {8},
  pages   = {1399--1413},
  year    = {2020},
  doi     = {10.2514/1.G004549}
}

% Continuous-time constraint satisfaction (CTCS)
@article{elangoContinuoustimeSuccessiveConvexification2025,
  title   = {Continuous-time successive convexification for constrained trajectory optimization},
  author  = {Elango, Purnanand and Luo, Dayou and Kamath, Abhinav G. and Uzun, Samet and Kim, Taewan and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  journal = {Automatica},
  volume  = {180},
  pages   = {112464},
  year    = {2025},
  doi     = {10.1016/j.automatica.2025.112464}
}

% Continuous-time compound state-triggered constraints (CT-cSTC)
@inproceedings{uzunSequentialConvexProgramming2025,
  title     = {Sequential Convex Programming for 6-{DoF} Powered Descent Guidance with Continuous-Time Compound State-Triggered Constraints},
  author    = {Uzun, Samet and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et and Carson, John M.},
  booktitle = {AIAA SciTech 2025 Forum},
  year      = {2025},
  doi       = {10.2514/6.2025-1895},
  note      = {Code: \url{https://github.com/UW-ACL/CT-cSTC}}
}
```

## Author and license
Author: mattnoh. There is no license file yet. Without one, others cannot legally reuse the
code. Add one (Aligator uses BSD-2) before publishing if you want it reused.
