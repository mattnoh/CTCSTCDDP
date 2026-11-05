/// @file stage_function.hpp
/// @brief Aligator adapter: a ctcs_stc::Spec as an aligator::StageFunctionTpl.
///
/// Adds no constraint math; it exposes the spec's rows c(x_k, u_k) <= 0 to Aligator.
/// Pair it with `aligator::NegativeOrthantTpl` (rows are <= 0), never an equality set:
///
///   ctcs_stc::Spec<double> spec(ctcs_stc::Form::Density, eps);
///   spec.add("...").triggerWhenNegative(g, dg).consequence(c, dcx, dcu);
///   spec.setIntervalDynamics(dyn);
///   stage.addConstraint(ctcs_stc::StageFunction(nx, nu, spec),
///                       aligator::NegativeOrthantTpl<double>());
///
/// Curvature: `computeVectorHessianProducts` returns the spec's Gauss-Newton term.
/// ProxDDP calls it only with `HessianApprox::EXACT`; under the default
/// `GAUSS_NEWTON` it is never called and the rows are first-order.
/// `Curvature::None` keeps them first-order under EXACT too.
///
/// Limits:
///   * Vector-space states only (aligator::VectorSpaceTpl): the spec differentiates
///     w.r.t. x itself, so ndx must equal nx.
///   * The spec is stored BY VALUE and Aligator copies a function per stage, so
///     each stage owns its own spec and evaluation cache (safe with
///     multithreaded stage evaluation). Editing the spec after building the
///     problem does not reach the stages. Capture lambda state by value.
///   * The spec caches on exact (x, u). If the interval dynamics read externals
///     that change between solves (a predicted target, say), rebuild the stages.
///
/// @author <AUTHOR>
#pragma once

#include "ctcs_stc/spec.hpp"

#include <aligator/core/function-abstract.hpp>

#include <stdexcept>
#include <utility>

namespace ctcs_stc {

enum class Curvature {
    GaussNewton,  ///< spec's lagrangianHessian (used only under HessianApprox::EXACT)
    None          ///< always zero
};

class StageFunction : public aligator::StageFunctionTpl<double> {
public:
    using Base = aligator::StageFunctionTpl<double>;
    using Data = aligator::StageFunctionDataTpl<double>;
    using ConstVectorRef = Base::ConstVectorRef;

    /// @param nx    state dimension (== ndx; vector spaces only)
    /// @param nu    control dimension
    /// @param spec  the authored constraint, copied
    StageFunction(int nx, int nu, Spec<double> spec, Curvature curvature = Curvature::GaussNewton)
        : Base(nx, nu, checkedDim(spec)), spec_(std::move(spec)), curvature_(curvature) {}

    void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, Data& data) const override {
        data.value_ = spec_.c(Eigen::VectorXd(x), Eigen::VectorXd(u));
    }

    void computeJacobians(const ConstVectorRef& x, const ConstVectorRef& u, Data& data) const override {
        const Eigen::VectorXd xv(x), uv(u);
        data.Jx_ = spec_.cx(xv, uv);
        data.Ju_ = spec_.cu(xv, uv);
    }

    void computeVectorHessianProducts(const ConstVectorRef& x, const ConstVectorRef& u,
                                      const ConstVectorRef& lbda, Data& data) const override {
        if (curvature_ == Curvature::None) {
            data.vhp_buffer_.setZero();
            return;
        }
        const Eigen::MatrixXd H =
            spec_.lagrangianHessian(Eigen::VectorXd(x), Eigen::VectorXd(u), Eigen::VectorXd(lbda));
        const int n = this->ndx1, m = this->nu;
        data.Hxx_ = H.topLeftCorner(n, n);
        data.Hxu_ = H.topRightCorner(n, m);
        data.Huu_ = H.bottomRightCorner(m, m);
    }

    const Spec<double>& spec() const { return spec_; }

private:
    static int checkedDim(const Spec<double>& s) {
        if (s.dim() <= 0)
            throw std::invalid_argument(
                "ctcs_stc::StageFunction: the spec has no rows; add implications before wrapping it");
        return s.dim();
    }

    Spec<double> spec_;
    Curvature curvature_;
};

}  // namespace ctcs_stc
