// Correctness checks for ctcs_stc::Spec and the Aligator adapter. Plain exit code.
//   1. cx, cu vs central differences: every Form, fixed step (1 and 4 substeps)
//      and free final time (time-dilation control, which exercises d/dTh).
//   2. Non-augmented == augmented: the Density row equals y(t_k+1) - y(t_k) - eps for
//      the accumulator y' = H integrated with the same RK4 steps.
//   3. Adapter == spec: values, Jacobians and vector-Hessian blocks.
#include "point_mass.hpp"

#include "ctcs_stc/stage_function.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace point_mass;
using ctcs_stc::Form;

namespace {
int failures = 0;

void check(bool ok, const std::string &what, double value) {
  std::printf("[%s] %-58s %.3e\n", ok ? "pass" : "FAIL", what.c_str(), value);
  if (!ok) ++failures;
}

// Three implications covering a nontrivial trigger, a control-dependent consequence
// and an always-on path constraint. u = [a (3)] or [a (3); Th] with free time.
ctcs_stc::Spec<double> makeSpec(Form form, int substeps, bool free_time) {
  const int nu = free_time ? NU + 1 : NU;
  ctcs_stc::Spec<double> spec(form, 1e-3);
  const Vec3 o(0.4, -0.2, 1.0);
  spec.add("below_h_slow")
      .triggerWhenNegative([](const V &x) { return x(2) - 1.5; },
                           [](const V &) { V g = V::Zero(NX); g(2) = 1; return g; })
      .consequence([](const V &x, const V &) { return x.tail<3>().squaredNorm() - 0.25; },
                   [](const V &x, const V &) { V g = V::Zero(NX); g.tail<3>() = 2 * x.tail<3>(); return g; },
                   [nu](const V &, const V &) { return V::Zero(nu).eval(); });
  spec.add("below_h_thrust")
      .triggerWhenNegative([](const V &x) { return x(2) - 1.5; },
                           [](const V &) { V g = V::Zero(NX); g(2) = 1; return g; })
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
      .triggerWhenNegative([](const V &) { return -1.0; }, [](const V &) { return V::Zero(NX).eval(); })
      .consequence([o](const V &x, const V &) { return 1.0 - (x.head<3>() - o).norm() / 0.8; },
                   [o](const V &x, const V &) {
                     const Vec3 d = x.head<3>() - o;
                     V g = V::Zero(NX);
                     g.head<3>() = -d / (0.8 * d.norm());
                     return g;
                   },
                   [nu](const V &, const V &) { return V::Zero(nu).eval(); });
  auto dyn = intervalDynamics(0.3, substeps);
  if (free_time) {
    dyn.idx_theta = NU;
    dyn.dt = 0.0;
  }
  spec.setIntervalDynamics(dyn);
  return spec;
}

// A point whose interval enters the trigger region, the keep-out ball and the thrust limit.
void point(bool free_time, V &x, V &u) {
  x.resize(NX);
  x << 0.1, 0.1, 1.7, 0.6, -0.5, -1.2;
  u.resize(free_time ? NU + 1 : NU);
  u.head<3>() << 2.0, -1.0, 13.0;
  if (free_time) u(NU) = 0.35;
}

double maxRelFdError(const ctcs_stc::Spec<double> &spec, const V &x, const V &u) {
  const M Jx = spec.cx(x, u), Ju = spec.cu(x, u);
  const double h = 1e-6;
  double err = 0.0, scale = 1e-12;
  for (int j = 0; j < x.size(); ++j) {
    V xp = x, xm = x;
    xp(j) += h;
    xm(j) -= h;
    const V fd = (spec.c(xp, u) - spec.c(xm, u)) / (2 * h);
    err = std::max(err, (fd - Jx.col(j)).cwiseAbs().maxCoeff());
    scale = std::max(scale, Jx.col(j).cwiseAbs().maxCoeff());
  }
  for (int j = 0; j < u.size(); ++j) {
    V up = u, um = u;
    up(j) += h;
    um(j) -= h;
    const V fd = (spec.c(x, up) - spec.c(x, um)) / (2 * h);
    err = std::max(err, (fd - Ju.col(j)).cwiseAbs().maxCoeff());
    scale = std::max(scale, Ju.col(j).cwiseAbs().maxCoeff());
  }
  return err / scale;
}

// RK4 on the augmented state [x; y], y' = H(x, u): the "augmented" CTCS of Elango 2025.
double accumulatorIncrement(const ctcs_stc::Spec<double> &spec, const V &x, const V &u, double Th, int M) {
  auto rhs = [&](const V &z) {
    V f(NX + 1);
    f.head(NX) << z.segment<3>(3), u.head<3>() + GRAVITY;
    f(NX) = spec.density(z.head(NX), u);
    return f;
  };
  V z(NX + 1);
  z << x, 0.0;
  const double h = Th / M;
  for (int j = 0; j < M; ++j) {
    const V k1 = rhs(z), k2 = rhs(z + 0.5 * h * k1), k3 = rhs(z + 0.5 * h * k2), k4 = rhs(z + h * k3);
    z += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
  }
  return z(NX);
}
}  // namespace

int main() {
  V x, u;
  // 1. Jacobians vs central differences.
  const struct { Form form; const char *name; } forms[] = {
      {Form::Original2018, "Original2018"}, {Form::Density, "Density"}, {Form::DensitySplit, "DensitySplit"}};
  const struct { int substeps; bool free_time; const char *name; } grids[] = {
      {1, false, "dt fixed, 1 substep"}, {4, false, "dt fixed, 4 substeps"}, {2, true, "free time, 2 substeps"}};
  for (const auto &f : forms)
    for (const auto &g : grids) {
      const auto spec = makeSpec(f.form, g.substeps, g.free_time);
      point(g.free_time, x, u);
      const bool active = spec.c(x, u).maxCoeff() > -spec.getEpsilon() + 1e-12 || f.form == Form::Original2018;
      const double e = maxRelFdError(spec, x, u);
      check(active && e < 1e-6, std::string("FD Jacobians, ") + f.name + ", " + g.name, e);
    }

  // 2. Non-augmented row == RK4 accumulator increment.
  for (const auto &g : grids) {
    const auto spec = makeSpec(Form::Density, g.substeps, g.free_time);
    point(g.free_time, x, u);
    const double Th = g.free_time ? u(NU) : 0.3;
    const double I = spec.c(x, u)(0) + spec.getEpsilon();
    const double y = accumulatorIncrement(spec, x, u, Th, g.substeps);
    const double rel = std::abs(I - y) / std::max(std::abs(y), 1e-300);
    check(y > 0.0 && rel < 1e-12, std::string("Density row == RK4 accumulator, ") + g.name, rel);
  }

  // 3. Adapter == spec.
  for (const auto &f : forms) {
    const auto spec = makeSpec(f.form, 4, false);
    point(false, x, u);
    const ctcs_stc::StageFunction fn(NX, NU, spec);
    auto data = fn.createData();
    fn.evaluate(x, u, *data);
    fn.computeJacobians(x, u, *data);
    const V lbda = V::Constant(spec.dim(), 0.7);
    fn.computeVectorHessianProducts(x, u, lbda, *data);
    const M H = spec.lagrangianHessian(x, u, lbda);
    const bool same = data->value_ == spec.c(x, u) && M(data->Jx_) == spec.cx(x, u) &&
                      M(data->Ju_) == spec.cu(x, u) && M(data->Hxx_) == H.topLeftCorner(NX, NX) &&
                      M(data->Hxu_) == H.topRightCorner(NX, NU) && M(data->Huu_) == H.bottomRightCorner(NU, NU);
    check(same, std::string("adapter == spec (value, Jx, Ju, vhp), ") + f.name, H.norm());

    const ctcs_stc::StageFunction first_order(NX, NU, spec, ctcs_stc::Curvature::None);
    auto d2 = first_order.createData();
    first_order.computeVectorHessianProducts(x, u, lbda, *d2);
    check(d2->vhp_buffer_.isZero(0.0), std::string("Curvature::None gives zero vhp, ") + f.name,
          d2->vhp_buffer_.norm());
  }

  std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
