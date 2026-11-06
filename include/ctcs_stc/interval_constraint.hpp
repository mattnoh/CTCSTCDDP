/// @file interval_constraint.hpp
/// @brief A Spec as Aligator stage rows c(x_k, u_k) <= 0, integrated along the
///        stage's own IntegratorRK4.
///
///   ctcs_stc::IntegratorRK4 dyn(MyODE(space, nu), dt);
///   ctcs_stc::Spec spec(ctcs_stc::Form::Density, eps);
///   spec.add("...").triggerWhenNegative(g, dg).consequence(c, dcx, dcu);
///   aligator::context::StageModel stage(cost, dyn);
///   stage.addConstraint(ctcs_stc::IntervalConstraint(spec, dyn),
///                       aligator::NegativeOrthantTpl<double>());
///
/// The constraint copies the integrator's ODE, step rule (fixed dt or FreeTime) and
/// substeps, and integrates the density at the same RK4 stage points with Simpson
/// weights.  The row therefore depends on (x_k, u_k) only and equals RK4 on an
/// accumulator y' = H over the same step (Elango et al. 2025, eq. 34).  The
/// integrator's projection is not applied inside the interval.
///
/// Curvature: `computeVectorHessianProducts` returns the Gauss-Newton term of the
/// rows.  ProxDDP calls it only with `HessianApprox::EXACT`; under the default
/// `GAUSS_NEWTON` the rows are first-order.  `Curvature::None` forces first order.
///
/// Per-stage buffers (ODE data, stage points, sensitivities) live in the
/// constraint's Data, created by Aligator per stage, so stages can be evaluated
/// in parallel.
///
/// @author mattnoh
#pragma once

#include "ctcs_stc/integrator_rk4.hpp"
#include "ctcs_stc/spec.hpp"

#include <aligator/core/function-abstract.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ctcs_stc {

enum class Curvature {
    GaussNewton,  ///< Gauss-Newton term of the rows (used only under HessianApprox::EXACT)
    None          ///< always zero
};

template <typename Scalar>
struct IntervalConstraintDataTpl;

template <typename _Scalar>
struct IntervalConstraintTpl : aligator::StageFunctionTpl<_Scalar> {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using BaseData = aligator::StageFunctionDataTpl<Scalar>;
    using Data = IntervalConstraintDataTpl<Scalar>;
    using ODEType = aligator::dynamics::ODEAbstractTpl<Scalar>;
    using Integrator = IntegratorRK4Tpl<Scalar>;
    using SpecType = SpecTpl<Scalar>;

    /// @param spec        the authored implications, copied
    /// @param integrator  the stage dynamics; its ODE, step rule and substeps are copied
    IntervalConstraintTpl(SpecType spec, const Integrator& integrator, Curvature curvature = Curvature::GaussNewton)
        : Base(integrator.ndx1(), integrator.nu, checkedDim(spec)),
          spec_(std::move(spec)),
          ode_(integrator.ode_),
          timestep_(integrator.timestep_),
          theta_index_(integrator.theta_index_),
          substeps_(integrator.substeps_),
          curvature_(curvature) {}

    const SpecType& spec() const { return spec_; }
    Scalar stepLength(const ConstVectorRef& u) const { return theta_index_ >= 0 ? u(theta_index_) : timestep_; }

    void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, BaseData& data) const override {
        Data& d = static_cast<Data&>(data);
        const VectorXs xv(x), uv(u);
        const int n = spec_.numImplications();
        if (spec_.form() == Form::Original2018) {
            for (int i = 0; i < n; ++i) d.value_(i) = spec_.residual2018(i, xv, uv);
            return;
        }
        detail::rk4Chain<Scalar>(*ode_, d.stages, xv, uv, stepLength(u), substeps_, true, false, d.value_chain);
        if (spec_.form() == Form::Density) {
            d.value_(0) = integral(d.value_chain, uv, -1) - spec_.epsilon();
        } else {
            for (int i = 0; i < n; ++i) d.value_(i) = integral(d.value_chain, uv, i) - spec_.epsilonOf(i);
        }
    }

    void computeJacobians(const ConstVectorRef& x, const ConstVectorRef& u, BaseData& data) const override {
        Data& d = static_cast<Data&>(data);
        const VectorXs xv(x), uv(u);
        const int n = spec_.numImplications();
        d.Jx_.setZero();
        d.Ju_.setZero();
        if (spec_.form() == Form::Original2018) {
            for (int i = 0; i < n; ++i) {
                const auto& im = spec_.implication(i);
                const Scalar g = im.trigger(xv), cv = spec_.consequenceVal(im, xv, uv);
                const Scalar a = SpecType::posPart(-g);
                // r = a c,  da/dx = -dg/dx where g < 0, else 0
                if (g < Scalar(0)) d.Jx_.row(i).noalias() += (-cv) * spec_.triggerDx(im, xv).transpose();
                d.Jx_.row(i).noalias() += a * spec_.consequenceDx(im, xv, uv).transpose();
                d.Ju_.row(i).noalias() += a * spec_.consequenceDu(im, xv, uv).transpose();
            }
            return;
        }
        sensitivityChain(xv, uv, d);
        if (spec_.form() == Form::Density) {
            const Grad gr = accumulate(d.chain, xv, uv, -1);
            d.Jx_.row(0) = gr.dx.transpose();
            d.Ju_.row(0) = gr.du.transpose();
        } else {
            for (int i = 0; i < n; ++i) {
                const Grad gr = accumulate(d.chain, xv, uv, i);
                d.Jx_.row(i) = gr.dx.transpose();
                d.Ju_.row(i) = gr.du.transpose();
            }
        }
    }

    /// Gauss-Newton curvature: for I = sum r_j^2 keep 2 sum lambda J_rj^T J_rj.
    void computeVectorHessianProducts(const ConstVectorRef& x, const ConstVectorRef& u, const ConstVectorRef& lbda,
                                      BaseData& data) const override {
        Data& d = static_cast<Data&>(data);
        d.vhp_buffer_.setZero();
        if (curvature_ == Curvature::None || spec_.form() == Form::Original2018) return;
        const VectorXs xv(x), uv(u);
        sensitivityChain(xv, uv, d);
        const auto& st = d.chain;
        if (!(st.Th > Scalar(0))) throw std::invalid_argument("ctcs_stc: CTCS curvature requires positive duration");
        const int nx = static_cast<int>(xv.size()), nu = static_cast<int>(uv.size());
        MatrixXs hessian = MatrixXs::Zero(nx + nu, nx + nu);
        for (const auto& sub : st.sub)
            for (const auto& pt : sub)
                for (int i = 0; i < spec_.numImplications(); ++i) {
                    const auto& im = spec_.implication(i);
                    const Scalar a = SpecType::posPart(-im.trigger(pt.p));
                    const Scalar b = SpecType::posPart(spec_.consequenceVal(im, pt.p, uv));
                    if (!(a > Scalar(0) && b > Scalar(0))) continue;
                    const VectorXs gp = -b * spec_.triggerDx(im, pt.p) + a * spec_.consequenceDx(im, pt.p, uv);
                    const VectorXs gu = a * spec_.consequenceDu(im, pt.p, uv);
                    VectorXs j(nx + nu);
                    j.head(nx) = pt.Sx.transpose() * gp;
                    j.tail(nu) = pt.Su.transpose() * gp;
                    j.tail(nu) += gu;
                    if (theta_index_ >= 0) j(nx + theta_index_) += gp.dot(pt.ST) + a * b / (Scalar(2) * st.Th);
                    const Scalar y = lbda(spec_.form() == Form::DensitySplit ? i : 0);
                    hessian.noalias() += (Scalar(2) * y * st.h * pt.w * im.weight) * (j * j.transpose());
                }
        d.vhp_buffer_ = hessian;  // full symmetric block (Hxx, Hxu, Hux, Huu)
    }

    std::shared_ptr<BaseData> createData() const override { return std::make_shared<Data>(*this); }

private:
    friend struct IntervalConstraintDataTpl<Scalar>;
    struct Grad {
        Scalar value = Scalar(0);
        VectorXs dx, du;
    };

    static int checkedDim(const SpecType& s) {
        if (s.dim() <= 0)
            throw std::invalid_argument("ctcs_stc::IntervalConstraint: the spec has no implications");
        return s.dim();
    }

    /// Value of the interval integral from the stage points (Simpson weights).
    Scalar integral(const detail::RK4Chain<Scalar>& st, const VectorXs& u, int imp) const {
        Scalar total = Scalar(0);
        for (const auto& sub : st.sub) {
            Scalar sum = Scalar(0);
            for (int i = 0; i < 4; ++i) sum += sub[i].w * spec_.density(sub[i].p, u, imp);
            total += st.h * sum;
        }
        return total;
    }

    /// Fill d.chain with sensitivities at (x, u), reusing it if already there.
    void sensitivityChain(const VectorXs& x, const VectorXs& u, Data& d) const {
        if (d.chain_valid && d.chain_x == x && d.chain_u == u) return;
        detail::rk4Chain<Scalar>(*ode_, d.stages, x, u, stepLength(u), substeps_, true, true, d.chain);
        d.chain_x = x;
        d.chain_u = u;
        d.chain_valid = true;
    }

    ///   dI/dx  = h sum_j sum_i w_i Sx_i^T Hx_i
    ///   dI/du  = h sum_j sum_i w_i (Su_i^T Hx_i + Hu_i)
    ///   dI/dTh = sum_j [ (1/M) sum_i w_i H_i + h sum_i w_i Hx_i . ST_i ]
    Grad accumulate(const detail::RK4Chain<Scalar>& st, const VectorXs& x, const VectorXs& u, int imp) const {
        Grad out;
        out.dx = VectorXs::Zero(x.size());
        out.du = VectorXs::Zero(u.size());
        Scalar dT = Scalar(0);
        for (const auto& sub : st.sub) {
            VectorXs sdx = VectorXs::Zero(x.size());
            VectorXs sdu = VectorXs::Zero(u.size());
            Scalar sum_h = Scalar(0), gT = Scalar(0);
            for (int i = 0; i < 4; ++i) {
                const auto& pt = sub[i];
                const VectorXs Hx = spec_.densityDx(pt.p, u, imp);
                const VectorXs Hu = spec_.densityDu(pt.p, u, imp);
                sum_h += pt.w * spec_.density(pt.p, u, imp);
                sdx.noalias() += pt.w * (pt.Sx.transpose() * Hx);
                sdu.noalias() += pt.w * (pt.Su.transpose() * Hx + Hu);
                gT += pt.w * Hx.dot(pt.ST);
            }
            sdx *= st.h;
            sdu *= st.h;
            out.dx += sdx;
            out.du += sdu;
            out.value += st.h * sum_h;
            dT += ((st.M == 1) ? sum_h : sum_h / Scalar(st.M)) + st.h * gT;
        }
        if (theta_index_ >= 0) out.du(theta_index_) += dT;
        return out;
    }

    SpecType spec_;
    xyz::polymorphic<ODEType> ode_;
    Scalar timestep_;
    int theta_index_;
    int substeps_;
    Curvature curvature_;
};

template <typename Scalar>
struct IntervalConstraintDataTpl : aligator::StageFunctionDataTpl<Scalar> {
    using Base = aligator::StageFunctionDataTpl<Scalar>;
    using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    detail::StageData<Scalar> stages;
    detail::RK4Chain<Scalar> value_chain;  ///< stage points only (evaluate)
    detail::RK4Chain<Scalar> chain;        ///< with sensitivities (Jacobians, curvature)
    VectorXs chain_x, chain_u;
    bool chain_valid = false;
    explicit IntervalConstraintDataTpl(const IntervalConstraintTpl<Scalar>& f)
        : Base(f), stages(*f.ode_) {}
};

using IntervalConstraint = IntervalConstraintTpl<double>;

}  // namespace ctcs_stc
