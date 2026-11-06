// Point mass shared by the examples and tests: x = [p; v] in R^6, u = commanded
// acceleration in R^3, gravity along -z.  Written once as an Aligator ODE; the
// stage dynamics are ctcs_stc::IntegratorRK4 on it (exact here: the flow is
// quadratic in time under a constant control).
#pragma once

#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/vector-space.hpp>
#include <aligator/modelling/dynamics/ode-abstract.hpp>

#include <Eigen/Dense>

namespace point_mass {

constexpr int NX = 6, NU = 3;
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;
using Vec3 = Eigen::Vector3d;
using Space = aligator::context::VectorSpace;
const Vec3 GRAVITY(0.0, 0.0, -9.81);

/// xdot = [v; u + g]
struct ODE : aligator::dynamics::ODEAbstractTpl<double> {
  using Base = aligator::dynamics::ODEAbstractTpl<double>;
  using Data = aligator::dynamics::ContinuousDynamicsDataTpl<double>;
  explicit ODE(const Space &space) : Base(space, NU) {}
  void forward(const ConstVectorRef &x, const ConstVectorRef &u, Data &d) const override {
    d.xdot_.head<3>() = x.tail<3>();
    d.xdot_.tail<3>() = u.head<3>() + GRAVITY;
  }
  void dForward(const ConstVectorRef &, const ConstVectorRef &, Data &d) const override {
    d.Jx_.setZero();
    d.Jx_.topRightCorner(3, 3).setIdentity();
    d.Ju_.setZero();
    d.Ju_.bottomLeftCorner(3, 3).setIdentity();
  }
};

/// r dt ||u - u_ref||^2
struct EffortCost : aligator::CostAbstractTpl<double> {
  using Base = aligator::CostAbstractTpl<double>;
  using CostData = aligator::CostDataAbstractTpl<double>;
  double w;
  V u_ref;
  EffortCost(const Space &space, double r_dt, const V &uref) : Base(space, NU), w(r_dt), u_ref(uref) {}
  void evaluate(const ConstVectorRef &, const ConstVectorRef &u, CostData &d) const override {
    d.value_ = w * (u - u_ref).squaredNorm();
  }
  void computeGradients(const ConstVectorRef &, const ConstVectorRef &u, CostData &d) const override {
    d.Lx_.setZero();
    d.Lu_ = 2.0 * w * (u - u_ref);
  }
  void computeHessians(const ConstVectorRef &, const ConstVectorRef &, CostData &d) const override {
    d.Lxx_.setZero();
    d.Lxu_.setZero();
    d.Luu_.setIdentity();
    d.Luu_ *= 2.0 * w;
  }
};

/// 0.5 W ||x - x_f||^2
struct TerminalCost : aligator::CostAbstractTpl<double> {
  using Base = aligator::CostAbstractTpl<double>;
  using CostData = aligator::CostDataAbstractTpl<double>;
  double W;
  V target;
  TerminalCost(const Space &space, double w, const V &xf) : Base(space, 0), W(w), target(xf) {}
  void evaluate(const ConstVectorRef &x, const ConstVectorRef &, CostData &d) const override {
    d.value_ = 0.5 * W * (x - target).squaredNorm();
  }
  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &, CostData &d) const override {
    d.Lx_ = W * (x - target);
    d.Lu_.setZero();
  }
  void computeHessians(const ConstVectorRef &, const ConstVectorRef &, CostData &d) const override {
    d.Lxx_.setIdentity();
    d.Lxx_ *= W;
    d.Lxu_.setZero();
    d.Luu_.setZero();
  }
};

/// Exact state at time s into an interval that starts at x with control u.
inline V flow(const V &x, const V &u, double s) {
  const Vec3 a = u.head<3>() + GRAVITY;
  V y(NX);
  y.head<3>() = x.head<3>() + s * x.tail<3>() + 0.5 * s * s * a;
  y.tail<3>() = x.tail<3>() + s * a;
  return y;
}

}  // namespace point_mass
