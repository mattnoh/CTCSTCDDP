/// @file integrator_rk4.hpp
/// @brief Classic RK4 explicit integrator for Aligator ODE models, with optional free
///        step length, substeps and a post-step projection.
///
/// Aligator ships Euler, semi-implicit Euler, midpoint and RK2.  The CTCS row
/// (interval_constraint.hpp) integrates its density along the RK4 stage points of
/// THIS integrator, which is what makes the row equal to RK4 on an accumulator
/// y' = H (Elango et al. 2025, eq. 34, non-augmented).  Both are built from the
/// same `RK4Chain`, so the discrete map and the row cannot disagree.
///
///   ctcs_stc::IntegratorRK4 dyn(MyODE(space, nu), dt);                 // fixed step
///   ctcs_stc::IntegratorRK4 dyn(MyODE(space, nu), ctcs_stc::FreeTime{i}); // h = u(i)
///
/// Vector-space states only (the stage updates are x + h k).  A projection, if set,
/// is applied ONCE after the full step (e.g. renormalising a quaternion block);
/// the stage points inside the interval are not projected.
///
/// @author <AUTHOR>
#pragma once

#include <aligator/modelling/dynamics/integrator-explicit.hpp>
#include <aligator/modelling/dynamics/ode-abstract.hpp>

#include <Eigen/Dense>

#include <array>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ctcs_stc {

/// Step length taken from a control component: h = u(index) (free final time).
struct FreeTime {
    int index;
};

/// x -> P(x) after the step, with its Jacobian at the unprojected state.
template <typename Scalar>
struct ProjectionTpl {
    using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    std::function<VectorXs(const VectorXs&)> value;
    std::function<MatrixXs(const VectorXs&)> jacobian;
    explicit operator bool() const { return static_cast<bool>(value); }
};

/// Renormalise x.segment(start, n) to unit norm (e.g. a quaternion block).
template <typename Scalar>
ProjectionTpl<Scalar> normalizeSegment(int start, int n) {
    using VectorXs = typename ProjectionTpl<Scalar>::VectorXs;
    using MatrixXs = typename ProjectionTpl<Scalar>::MatrixXs;
    ProjectionTpl<Scalar> p;
    p.value = [start, n](const VectorXs& x) {
        VectorXs y = x;
        y.segment(start, n).normalize();
        return y;
    };
    p.jacobian = [start, n](const VectorXs& x) {
        const VectorXs q = x.segment(start, n);
        const Scalar r = q.norm();
        MatrixXs J = MatrixXs::Identity(x.size(), x.size());
        J.block(start, start, n, n) = (MatrixXs::Identity(n, n) - q * q.transpose() / (r * r)) / r;
        return J;
    };
    return p;
}

namespace detail {

/// One RK4 stage point with its sensitivities to the interval inputs (x, u, Th).
template <typename Scalar>
struct StagePoint {
    using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    VectorXs p;   ///< stage point
    MatrixXs Sx;  ///< d p / d x
    MatrixXs Su;  ///< d p / d u
    VectorXs ST;  ///< d p / d Th
    Scalar w;     ///< Simpson weight, (1,2,2,1)/6
};

/// The RK4 interval: stage points of every substep and the end state, optionally
/// with their sensitivities.  End state is BEFORE any projection.
template <typename Scalar>
struct RK4Chain {
    using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Scalar Th = Scalar(0), h = Scalar(0);
    int M = 1;
    std::vector<std::array<StagePoint<Scalar>, 4>> sub;
    VectorXs xend;
    MatrixXs Jx, Ju;  ///< d xend / d x, d xend / d u  (u direct, without the Th column)
    VectorXs JT;      ///< d xend / d Th
};

/// ODE data objects for the four RK4 stages, reused across substeps.
template <typename Scalar>
struct StageData {
    using ODEData = aligator::dynamics::ContinuousDynamicsDataTpl<Scalar>;
    std::array<std::shared_ptr<ODEData>, 4> d;
    explicit StageData(const aligator::dynamics::ODEAbstractTpl<Scalar>& ode) {
        for (auto& p : d) p = ode.createData();
    }
};

/// Integrate one interval of length Th in M RK4 substeps along `ode`.
///   points: fill chain.sub (stage points);  sens: fill all sensitivities.
template <typename Scalar>
void rk4Chain(const aligator::dynamics::ODEAbstractTpl<Scalar>& ode, StageData<Scalar>& sd,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& u, Scalar Th, int M, bool points,
              bool sens, RK4Chain<Scalar>& s) {
    using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int nx = static_cast<int>(x.size()), nu = static_cast<int>(u.size());
    s.Th = Th;
    s.M = M;
    s.h = Th / Scalar(M);
    const Scalar h = s.h, h2 = Scalar(0.5) * h;
    static const Scalar kW[4] = {Scalar(1) / Scalar(6), Scalar(2) / Scalar(6), Scalar(2) / Scalar(6),
                                 Scalar(1) / Scalar(6)};
    auto& D1 = *sd.d[0];
    auto& D2 = *sd.d[1];
    auto& D3 = *sd.d[2];
    auto& D4 = *sd.d[3];
    const MatrixXs I = MatrixXs::Identity(nx, nx);
    if (points) s.sub.resize(static_cast<size_t>(M));

    VectorXs xj = x;
    MatrixXs Phi, Psi;  // d xj / d x, d xj / d u
    VectorXs psiT;      // d xj / d Th
    for (int j = 0; j < M; ++j) {
        ode.forward(xj, u, D1);
        const VectorXs k1 = D1.xdot_;
        const VectorXs x2 = xj + h2 * k1;
        ode.forward(x2, u, D2);
        const VectorXs k2 = D2.xdot_;
        const VectorXs x3 = xj + h2 * k2;
        ode.forward(x3, u, D3);
        const VectorXs k3 = D3.xdot_;
        const VectorXs x4 = xj + h * k3;
        ode.forward(x4, u, D4);
        const VectorXs k4 = D4.xdot_;

        if (!sens) {
            if (points) {
                const VectorXs P[4] = {xj, x2, x3, x4};
                for (int i = 0; i < 4; ++i) {
                    s.sub[static_cast<size_t>(j)][i].p = P[i];
                    s.sub[static_cast<size_t>(j)][i].w = kW[i];
                }
            }
            xj = xj + (h / Scalar(6)) * (k1 + Scalar(2) * k2 + Scalar(2) * k3 + k4);
            continue;
        }

        ode.dForward(xj, u, D1);
        ode.dForward(x2, u, D2);
        ode.dForward(x3, u, D3);
        ode.dForward(x4, u, D4);
        const MatrixXs &A1 = D1.Jx_, &B1 = D1.Ju_, &A2 = D2.Jx_, &B2 = D2.Ju_;
        const MatrixXs &A3 = D3.Jx_, &B3 = D3.Ju_, &A4 = D4.Jx_, &B4 = D4.Ju_;

        // local sensitivities, w.r.t. the substep's own start state and h
        const MatrixXs d1x = A1, d1u = B1;
        const MatrixXs d2x = A2 * (I + h2 * d1x);
        const MatrixXs d2u = A2 * (h2 * d1u) + B2;
        const VectorXs d2T = A2 * (Scalar(0.5) * k1);
        const MatrixXs d3x = A3 * (I + h2 * d2x);
        const MatrixXs d3u = A3 * (h2 * d2u) + B3;
        const VectorXs d3T = A3 * (Scalar(0.5) * k2 + h2 * d2T);

        const MatrixXs LSx[4] = {I, I + h2 * d1x, I + h2 * d2x, I + h * d3x};
        const MatrixXs LSu[4] = {MatrixXs::Zero(nx, nu), h2 * d1u, h2 * d2u, h * d3u};
        const VectorXs LST[4] = {VectorXs::Zero(nx), Scalar(0.5) * k1, Scalar(0.5) * k2 + h2 * d2T,
                                 k3 + h * d3T};

        if (points) {
            const VectorXs P[4] = {xj, x2, x3, x4};
            auto& pts = s.sub[static_cast<size_t>(j)];
            for (int i = 0; i < 4; ++i) {
                auto& pt = pts[i];
                pt.p = P[i];
                pt.w = kW[i];
                if (j == 0) {
                    pt.Sx = LSx[i];
                    pt.Su = LSu[i];
                    pt.ST = (M == 1) ? LST[i] : VectorXs(LST[i] / Scalar(M));
                } else {
                    pt.Sx = LSx[i] * Phi;
                    pt.Su = LSx[i] * Psi + LSu[i];
                    pt.ST = LSx[i] * psiT + LST[i] / Scalar(M);
                }
            }
        }

        // Full RK4 step of this substep and its chain.
        const MatrixXs d4x = A4 * LSx[3];
        const MatrixXs d4u = A4 * LSu[3] + B4;
        const VectorXs d4T = A4 * LST[3];
        const MatrixXs Jx = I + (h / Scalar(6)) * (d1x + Scalar(2) * d2x + Scalar(2) * d3x + d4x);
        const MatrixXs Ju = (h / Scalar(6)) * (d1u + Scalar(2) * d2u + Scalar(2) * d3u + d4u);
        const VectorXs Jh = (k1 + Scalar(2) * k2 + Scalar(2) * k3 + k4) / Scalar(6) +
                            (h / Scalar(6)) * (Scalar(2) * d2T + Scalar(2) * d3T + d4T);
        if (j == 0) {
            Phi = Jx;
            Psi = Ju;
            psiT = Jh / Scalar(M);
        } else {
            Phi = (Jx * Phi).eval();
            Psi = (Jx * Psi + Ju).eval();
            psiT = (Jx * psiT + Jh / Scalar(M)).eval();
        }
        xj = xj + (h / Scalar(6)) * (k1 + Scalar(2) * k2 + Scalar(2) * k3 + k4);
    }
    s.xend = xj;
    if (sens) {
        s.Jx = Phi;
        s.Ju = Psi;
        s.JT = psiT;
    }
}

}  // namespace detail

template <typename Scalar>
struct IntegratorRK4DataTpl;

/// x_{k+1} = P(RK4^M(x_k, u_k, h)), h fixed or h = u(FreeTime::index).
template <typename _Scalar>
struct IntegratorRK4Tpl : aligator::dynamics::ExplicitIntegratorAbstractTpl<_Scalar> {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::dynamics::ExplicitIntegratorAbstractTpl<Scalar>;
    using BaseData = aligator::ExplicitDynamicsDataTpl<Scalar>;
    using Data = IntegratorRK4DataTpl<Scalar>;
    using ODEType = typename Base::ODEType;
    using Projection = ProjectionTpl<Scalar>;

    Scalar timestep_ = Scalar(0);  ///< fixed step (theta_index_ < 0)
    int theta_index_ = -1;         ///< control index of the step length, or -1
    int substeps_ = 1;             ///< RK4 substeps per interval
    Projection projection_;        ///< optional, after the full step

    IntegratorRK4Tpl(const xyz::polymorphic<ODEType>& ode, Scalar timestep, int substeps = 1,
                     Projection projection = {})
        : Base(ode), timestep_(timestep), substeps_(substeps), projection_(std::move(projection)) {
        check();
    }
    IntegratorRK4Tpl(const xyz::polymorphic<ODEType>& ode, FreeTime free_time, int substeps = 1,
                     Projection projection = {})
        : Base(ode), theta_index_(free_time.index), substeps_(substeps), projection_(std::move(projection)) {
        check();
    }

    /// Interval length at control u.
    Scalar stepLength(const ConstVectorRef& u) const { return theta_index_ >= 0 ? u(theta_index_) : timestep_; }

    void forward(const ConstVectorRef& x, const ConstVectorRef& u, BaseData& data) const override {
        Data& d = static_cast<Data&>(data);
        detail::rk4Chain<Scalar>(*this->ode_, d.stages, VectorXs(x), VectorXs(u), stepLength(u), substeps_,
                                 false, false, d.chain);
        d.xnext_ = projection_ ? projection_.value(d.chain.xend) : d.chain.xend;
    }

    void dForward(const ConstVectorRef& x, const ConstVectorRef& u, BaseData& data) const override {
        Data& d = static_cast<Data&>(data);
        detail::rk4Chain<Scalar>(*this->ode_, d.stages, VectorXs(x), VectorXs(u), stepLength(u), substeps_,
                                 false, true, d.chain);
        MatrixXs Jx = d.chain.Jx, Ju = d.chain.Ju;
        if (theta_index_ >= 0) Ju.col(theta_index_) += d.chain.JT;
        if (projection_) {
            const MatrixXs N = projection_.jacobian(d.chain.xend);
            d.Jx() = N * Jx;
            d.Ju() = N * Ju;
        } else {
            d.Jx() = Jx;
            d.Ju() = Ju;
        }
    }

    std::shared_ptr<BaseData> createData() const override { return std::make_shared<Data>(*this); }

private:
    void check() const {
        if (this->space_next_->nx() != this->space_next_->ndx())
            throw std::invalid_argument("ctcs_stc::IntegratorRK4: vector-space states only (nx == ndx)");
        if (substeps_ < 1) throw std::invalid_argument("ctcs_stc::IntegratorRK4: substeps must be >= 1");
        if (theta_index_ < 0 && !(timestep_ > Scalar(0)))
            throw std::invalid_argument("ctcs_stc::IntegratorRK4: fixed step must be positive");
        if (theta_index_ >= this->nu)
            throw std::invalid_argument("ctcs_stc::IntegratorRK4: FreeTime index out of range");
    }
};

template <typename Scalar>
struct IntegratorRK4DataTpl : aligator::dynamics::ExplicitIntegratorDataTpl<Scalar> {
    using Base = aligator::dynamics::ExplicitIntegratorDataTpl<Scalar>;
    detail::StageData<Scalar> stages;
    detail::RK4Chain<Scalar> chain;
    explicit IntegratorRK4DataTpl(const IntegratorRK4Tpl<Scalar>& integrator)
        : Base(integrator), stages(*integrator.ode_) {}
};

using IntegratorRK4 = IntegratorRK4Tpl<double>;
using Projection = ProjectionTpl<double>;

}  // namespace ctcs_stc
