// Point mass shared by the examples and tests: x = [p; v] in R^6, u = commanded
// acceleration in R^3, gravity along -z.  The discrete map is the exact flow for a
// control held constant over dt, and `intervalDynamics` is its continuous
// counterpart: the spec must integrate the SAME model the solver steps.
#pragma once

#include "ctcs_stc/spec.hpp"

#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/explicit-dynamics.hpp>
#include <aligator/core/vector-space.hpp>

#include <Eigen/Dense>

namespace point_mass {

constexpr int NX = 6, NU = 3;
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;
using Vec3 = Eigen::Vector3d;
using Space = aligator::context::VectorSpace;
const Vec3 GRAVITY(0.0, 0.0, -9.81);

struct Dynamics : aligator::ExplicitDynamicsModelTpl<double> {
  using Base = aligator::ExplicitDynamicsModelTpl<double>;
  using Data = aligator::ExplicitDynamicsDataTpl<double>;
  double dt;
  Dynamics(const Space &space, double dt_) : Base(space, NU), dt(dt_) {}
  void forward(const ConstVectorRef &x, const ConstVectorRef &u, Data &data) const override {
    const Vec3 a = u + GRAVITY;
    data.xnext_.head<3>() = x.head<3>() + dt * x.tail<3>() + 0.5 * dt * dt * a;
    data.xnext_.tail<3>() = x.tail<3>() + dt * a;
  }
  void dForward(const ConstVectorRef &, const ConstVectorRef &, Data &data) const override {
    data.Jx().setIdentity();
    data.Jx().topRightCorner<3, 3>() = dt * Eigen::Matrix3d::Identity();
    data.Ju().setZero();
    data.Ju().topRows<3>() = 0.5 * dt * dt * Eigen::Matrix3d::Identity();
    data.Ju().bottomRows<3>() = dt * Eigen::Matrix3d::Identity();
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

/// Continuous dynamics xdot = [v; u + g] for the spec, fixed step dt.
inline ctcs_stc::IntervalDynamics<double> intervalDynamics(double dt, int substeps) {
  ctcs_stc::IntervalDynamics<double> d;
  d.xdot = [](const V &x, const V &u, double) {
    V f(NX);
    f.head<3>() = x.tail<3>();
    f.tail<3>() = u.head<3>() + GRAVITY;
    return f;
  };
  d.fx = [](const V &, const V &, double) {
    M A = M::Zero(NX, NX);
    A.topRightCorner(3, 3).setIdentity();
    return A;
  };
  d.fu = [](const V &, const V &u, double) {
    M B = M::Zero(NX, u.size());
    B.bottomLeftCorner(3, 3).setIdentity();
    return B;
  };
  d.idx_theta = -1;
  d.dt = dt;
  d.substeps = substeps;
  return d;
}

/// Exact state at time s into an interval that starts at x with control u.
inline V flow(const V &x, const V &u, double s) {
  const Vec3 a = u.head<3>() + GRAVITY;
  V y(NX);
  y.head<3>() = x.head<3>() + s * x.tail<3>() + 0.5 * s * s * a;
  y.tail<3>() = x.tail<3>() + s * a;
  return y;
}

}  // namespace point_mass
