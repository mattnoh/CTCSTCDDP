// Crazyflie-class quadrotor with free step length, for examples/stc_landing.cpp.
//
//   x = [r (3), v (3), q = (qw, qx, qy, qz) (4), omega_B (3), t]   in R^14
//   u = [f~ (scaled thrust), M~ (3 scaled body moments), theta (step length)]  in R^5
//
// The solver sees controls in scaled units, [f_z, M] = U_CENTER + U_SCALE .* [f~, M~]
// (thrust about hover, both O(1) at the optimum); theta stays in seconds.
//
//   r' = v,  v' = C(q) e_z f_z / m + g,  q' = 0.5 Omega(omega) q,
//   omega' = J^-1 (M - omega x J omega),  t' = 1 (t_{k+1} = t_k + theta_k)
//
// One RK4 step of length theta per interval, quaternion renormalized after the
// step; Jacobians by RK4 forward sensitivity including the renormalization.
// The model, constants and Jacobians follow the single-horizon landing OCP this
// example reproduces (Quad6DOFVarTime, target acceleration zero).
#pragma once

#include <aligator/core/explicit-dynamics.hpp>
#include <aligator/core/vector-space.hpp>

#include <Eigen/Dense>

namespace quadrotor {

constexpr int NX = 14, NU = 5, NXP = 13, NUP = 4;
constexpr int IDX_T = 13, IDX_THETA = 4;
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

constexpr double MASS = 0.027, IXX = 1.66e-5, IYY = 1.66e-5, IZZ = 2.92e-5;
constexpr double J_SCALE = 1.0 / IXX;  // inertia (and moments) in units of Ixx
const Vec3 GRAVITY(0.0, 0.0, -9.81);
inline Mat3 inertia() { return Vec3(IXX * J_SCALE, IYY * J_SCALE, IZZ * J_SCALE).asDiagonal(); }

inline Mat3 rotation(const Eigen::Vector4d &q) {
  const double w = q(0), x = q(1), y = q(2), z = q(3);
  Mat3 C;
  C << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
       2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
       2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
  return C;
}
inline Eigen::Matrix4d omegaMatrix(const Vec3 &w) {
  Eigen::Matrix4d O;
  O << 0, -w(0), -w(1), -w(2),
       w(0), 0, w(2), -w(1),
       w(1), -w(2), 0, w(0),
       w(2), w(1), -w(0), 0;
  return O;
}
inline Mat3 skew(const Vec3 &v) {
  Mat3 S;
  S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return S;
}

/// Continuous rates of the 13 physical states (time excluded).
inline V xdot(const V &x, const V &u) {
  static const Mat3 J = inertia(), Jinv = J.inverse();
  const Vec3 w = x.segment<3>(10);
  V d(NXP);
  d.segment<3>(0) = x.segment<3>(3);
  d.segment<3>(3) = rotation(x.segment<4>(6)).col(2) * u(0) / MASS + GRAVITY;
  d.segment<4>(6) = 0.5 * omegaMatrix(w) * x.segment<4>(6);
  d.segment<3>(10) = Jinv * (u.segment<3>(1) - w.cross(J * w));
  return d;
}

/// d xdot / d x (13 x 13).
inline M A(const V &x, const V &u) {
  static const Mat3 J = inertia(), Jinv = J.inverse();
  const Eigen::Vector4d q = x.segment<4>(6);
  const Vec3 w = x.segment<3>(10);
  const double s = u(0) / MASS;
  M a = M::Zero(NXP, NXP);
  a.block<3, 3>(0, 3).setIdentity();
  // d (C e_z) / d q
  a.block<3, 1>(3, 6) = s * Vec3(2 * q(2), -2 * q(1), 0);
  a.block<3, 1>(3, 7) = s * Vec3(2 * q(3), -2 * q(0), -4 * q(1));
  a.block<3, 1>(3, 8) = s * Vec3(2 * q(0), 2 * q(3), -4 * q(2));
  a.block<3, 1>(3, 9) = s * Vec3(2 * q(1), 2 * q(2), 0);
  a.block<4, 4>(6, 6) = 0.5 * omegaMatrix(w);
  Eigen::Matrix<double, 4, 3> dq;
  dq << -q(1), -q(2), -q(3), q(0), -q(3), q(2), q(3), q(0), -q(1), -q(2), q(1), q(0);
  a.block<4, 3>(6, 10) = 0.5 * dq;
  a.block<3, 3>(10, 10) = Jinv * (skew(J * w) - skew(w) * J);
  return a;
}

/// d xdot / d [f_z, M] (13 x 4).
inline M B(const V &x) {
  static const Mat3 Jinv = inertia().inverse();
  M b = M::Zero(NXP, NUP);
  b.block<3, 1>(3, 0) = rotation(x.segment<4>(6)).col(2) / MASS;
  b.block<3, 3>(10, 1) = Jinv;
  return b;
}

/// Control scaling: physical [f_z, M] = U_CENTER + U_SCALE .* [f~, M~].
constexpr double F_SCALE = 0.05, M_SCALE = 0.1;
inline const V &U_CENTER() { static const V c = (V(NUP) << MASS * 9.81, 0, 0, 0).finished(); return c; }
inline const V &U_SCALE() { static const V s = (V(NUP) << F_SCALE, M_SCALE, M_SCALE, M_SCALE).finished(); return s; }
/// Physical [f_z, M] from a solver control u (theta ignored).
inline V physical(const V &u) { return U_CENTER() + U_SCALE().cwiseProduct(u.head(NUP)); }

/// x_{k+1} = RK4 step of length theta, quaternion renormalized, t += theta.
struct Dynamics : aligator::ExplicitDynamicsModelTpl<double> {
  using Base = aligator::ExplicitDynamicsModelTpl<double>;
  using Data = aligator::ExplicitDynamicsDataTpl<double>;
  explicit Dynamics(const aligator::context::VectorSpace &space) : Base(space, NU) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u, Data &data) const override {
    const V xp = x.head(NXP), up = physical(u);
    const double h = u(IDX_THETA);
    const V k1 = xdot(xp, up), k2 = xdot(xp + 0.5 * h * k1, up), k3 = xdot(xp + 0.5 * h * k2, up),
            k4 = xdot(xp + h * k3, up);
    V xn = xp + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
    xn.segment<4>(6).normalize();
    data.xnext_.head(NXP) = xn;
    data.xnext_(IDX_T) = x(IDX_T) + h;
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u, Data &data) const override {
    const V xp = x.head(NXP), up = physical(u);
    const double h = u(IDX_THETA);
    const M I = M::Identity(NXP, NXP);
    const V k1 = xdot(xp, up), x2 = xp + 0.5 * h * k1, k2 = xdot(x2, up), x3 = xp + 0.5 * h * k2,
            k3 = xdot(x3, up), x4 = xp + h * k3, k4 = xdot(x4, up);
    const M A1 = A(xp, up), A2 = A(x2, up), A3 = A(x3, up), A4 = A(x4, up);
    const M B1 = B(xp), B2 = B(x2), B3 = B(x3), B4 = B(x4);

    const M d1x = A1, d2x = A2 * (I + 0.5 * h * d1x), d3x = A3 * (I + 0.5 * h * d2x), d4x = A4 * (I + h * d3x);
    const M d1u = B1, d2u = A2 * (0.5 * h * d1u) + B2, d3u = A3 * (0.5 * h * d2u) + B3, d4u = A4 * (h * d3u) + B4;
    const V d2T = A2 * (0.5 * k1), d3T = A3 * (0.5 * k2 + 0.5 * h * d2T), d4T = A4 * (k3 + h * d3T);
    const M Jx = I + h / 6.0 * (d1x + 2 * d2x + 2 * d3x + d4x);
    const M Ju = h / 6.0 * (d1u + 2 * d2u + 2 * d3u + d4u);
    const V JT = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0 + h / 6.0 * (2 * d2T + 2 * d3T + d4T);

    // Renormalization q -> q / |q| after the step.
    const Eigen::Vector4d qr = (xp + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)).segment<4>(6);
    const double n = qr.norm();
    M N = I;
    N.block<4, 4>(6, 6) = (Eigen::Matrix4d::Identity() - qr * qr.transpose() / (n * n)) / n;

    data.Jx().setZero();
    data.Jx().topLeftCorner(NXP, NXP) = N * Jx;
    data.Jx()(IDX_T, IDX_T) = 1.0;
    data.Ju().setZero();
    data.Ju().topLeftCorner(NXP, NUP) = N * Ju * U_SCALE().asDiagonal();
    data.Ju().block(0, IDX_THETA, NXP, 1) = N * JT;
    data.Ju()(IDX_T, IDX_THETA) = 1.0;
  }
};

}  // namespace quadrotor
