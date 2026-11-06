/// @file spec.hpp
/// @brief Authoring of state-triggered (STC) and continuous-time (CTCS) constraints.
///
/// Convention (Szmuk et al. 2018; Uzun et al. 2025 eq. 1):
///
///     g(x) < 0   ==>   c(x,u) <= 0
///
/// The trigger is a function of the STATE ONLY; the consequence may use state and
/// control.  The density activates on [-g]_+, not [g]_+, which is why the setter is
/// named `triggerWhenNegative`.  A plain path constraint c(x,u) <= 0 is the special
/// case of an always-on trigger, g = -1.
///
/// A Spec holds the implications and the transcription (Form).  It does no
/// integration: `IntervalConstraint` (interval_constraint.hpp) turns it into
/// Aligator stage rows, integrating along the same `IntegratorRK4` the stage
/// dynamics use.
///
///   Original2018  one row per implication at the node:  [-g_i]_+ * c_i <= 0
///   Density       one row per interval:   I_k - eps <= 0
///   DensitySplit  one row per implication per interval:  I_{k,i} - eps_i <= 0
///
///   I_k = integral over [t_k, t_k+1] of  H(x(t),u_k),  H = sum_i w_i [-g_i]_+^2 [s_i c_i]_+^2
///
/// SCALING.  The Density form adds every implication into one row against one eps,
/// so write each consequence as a DIMENSIONLESS FRACTION OF ITS OWN LIMIT
/// (`scale(1/limit)`, or divide inside the lambda) and keep weights at 1 unless you
/// want a deliberate priority (Elango et al. 2025, Sec. 6):
///     obstacle |p - o| >= r   ->  1 - |p - o| / r
///     input    |a| <= a_max   ->  |a| / a_max - 1
/// eps is then a squared fractional violation and means the same for every row.
/// w [c]_+^2 == [sqrt(w) c]_+^2, so a weight is a scale written in the other place.
///
/// WHY NO AUTODIFF.  An Eigen::AutoDiffScalar path was written and removed: its
/// derivative expression templates outlive their temporaries, and an expression as
/// ordinary as  sqrt(x0*x0 + x1*x1 + 1e-12) - k*(x2 + b)  returned
/// nondeterministically wrong derivatives (ASan: heap-use-after-free inside Eigen).
/// Supply analytic Jacobians, or opt in to central differences per implication
/// with `.finiteDifference()`.  A missing Jacobian without that opt-in throws.
///
/// @author mattnoh
#pragma once

#include <Eigen/Dense>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctcs_stc {

/// Which published transcription emits the rows.
enum class Form {
    Original2018,  ///< nodal: [-g]_+ * c <= 0, one row per implication (Szmuk 2018)
    Density,       ///< interval: integral of [-g]_+^2 [c]_+^2 over the step - eps <= 0
    DensitySplit   ///< interval, one row per implication: I_i - eps_i <= 0
};

/// One implication `g(x) < 0 ==> c(x,u) <= 0`.
template <typename Scalar>
struct ImplicationTpl {
    using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    std::string name;

    /// g(x): fires where NEGATIVE.
    std::function<Scalar(const VectorXs&)> trigger;
    /// c(x,u): satisfied where <= 0.
    std::function<Scalar(const VectorXs&, const VectorXs&)> consequence;

    /// Analytic gradients (as column vectors).  Empty => error unless allow_fd.
    std::function<VectorXs(const VectorXs&)> trigger_dx;
    std::function<VectorXs(const VectorXs&, const VectorXs&)> consequence_dx;
    std::function<VectorXs(const VectorXs&, const VectorXs&)> consequence_du;

    /// Constraint scale s_i: the consequence is used as s_i * c_i everywhere.
    Scalar scale = Scalar(1);
    /// Density weight w_i.  Unused by Original2018.
    Scalar weight = Scalar(1);
    /// Per-implication budget for DensitySplit.  Negative => the spec-wide eps.
    Scalar epsilon = Scalar(-1);
    /// Central differences for the gradients this implication did not supply.
    bool allow_fd = false;
};

template <typename _Scalar>
class SpecTpl {
public:
    using Scalar = _Scalar;
    using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Implication = ImplicationTpl<Scalar>;

    /// @param form  which transcription emits the rows
    /// @param eps   interval budget (the CTCS epsilon-relaxation); ignored by Original2018
    explicit SpecTpl(Form form = Form::Density, Scalar eps = Scalar(0)) : form_(form), eps_(eps) {}

    // ── Authoring (chainable) ────────────────────────────────────────────────
    //   spec.add("landing_speed")
    //       .triggerWhenNegative(g, dg)
    //       .consequence(c, dcx, dcu)
    //       .scale(1.0 / V_MAX);

    SpecTpl& add(std::string name) {
        Implication impl;
        impl.name = std::move(name);
        impls_.push_back(std::move(impl));
        return *this;
    }
    SpecTpl& triggerWhenNegative(std::function<Scalar(const VectorXs&)> g,
                                 std::function<VectorXs(const VectorXs&)> dg_dx = {}) {
        last().trigger = std::move(g);
        last().trigger_dx = std::move(dg_dx);
        return *this;
    }
    SpecTpl& consequence(std::function<Scalar(const VectorXs&, const VectorXs&)> c,
                         std::function<VectorXs(const VectorXs&, const VectorXs&)> dc_dx = {},
                         std::function<VectorXs(const VectorXs&, const VectorXs&)> dc_du = {}) {
        last().consequence = std::move(c);
        last().consequence_dx = std::move(dc_dx);
        last().consequence_du = std::move(dc_du);
        return *this;
    }
    /// Scale this consequence by s (pass 1/limit); the zero set is unchanged.
    SpecTpl& scale(Scalar s) { last().scale = s; return *this; }
    /// Density weight; see SCALING above before using it.
    SpecTpl& weight(Scalar w) { last().weight = w; return *this; }
    /// Per-implication budget for Form::DensitySplit.
    SpecTpl& epsilon(Scalar e) { last().epsilon = e; return *this; }
    /// Accept central differences for this implication's missing gradients.
    SpecTpl& finiteDifference() { last().allow_fd = true; return *this; }

    // ── Configuration ────────────────────────────────────────────────────────
    void setForm(Form form) { form_ = form; }
    Form form() const { return form_; }
    void setEpsilon(Scalar eps) { eps_ = eps; }
    Scalar epsilon() const { return eps_; }
    void setFdStep(double h) { fd_h_ = h; }
    int numImplications() const { return static_cast<int>(impls_.size()); }
    const Implication& implication(int i) const { return impls_[static_cast<size_t>(i)]; }
    /// Rows the constraint emits: 1 for Density, one per implication otherwise.
    int dim() const {
        if (impls_.empty()) return 0;
        return form_ == Form::Density ? 1 : numImplications();
    }
    /// Budget of implication i (DensitySplit): its own eps, else the shared one.
    Scalar epsilonOf(int i) const {
        const Scalar e = implication(i).epsilon;
        return e >= Scalar(0) ? e : eps_;
    }

    // ── Pointwise kernel ─────────────────────────────────────────────────────
    /// sum_i w_i [-g_i]_+^2 [s_i c_i]_+^2 (imp < 0: all implications, else only imp).
    Scalar density(const VectorXs& x, const VectorXs& u, int imp = -1) const {
        Scalar h = Scalar(0);
        for (int i = 0; i < numImplications(); ++i) {
            if (imp >= 0 && i != imp) continue;
            const auto& im = implication(i);
            const Scalar a = posPart(-im.trigger(x));
            const Scalar b = posPart(consequenceVal(im, x, u));
            h += im.weight * a * a * b * b;
        }
        return h;
    }
    VectorXs densityDx(const VectorXs& x, const VectorXs& u, int imp = -1) const {
        VectorXs d = VectorXs::Zero(x.size());
        for (int i = 0; i < numImplications(); ++i) {
            if (imp >= 0 && i != imp) continue;
            const auto& im = implication(i);
            const Scalar a = posPart(-im.trigger(x)), b = posPart(consequenceVal(im, x, u));
            if (a <= Scalar(0) || b <= Scalar(0)) continue;
            d.noalias() += (im.weight * Scalar(-2) * a * b * b) * triggerDx(im, x);
            d.noalias() += (im.weight * Scalar(2) * a * a * b) * consequenceDx(im, x, u);
        }
        return d;
    }
    VectorXs densityDu(const VectorXs& x, const VectorXs& u, int imp = -1) const {
        VectorXs d = VectorXs::Zero(u.size());
        for (int i = 0; i < numImplications(); ++i) {
            if (imp >= 0 && i != imp) continue;
            const auto& im = implication(i);
            const Scalar a = posPart(-im.trigger(x)), b = posPart(consequenceVal(im, x, u));
            if (a <= Scalar(0) || b <= Scalar(0)) continue;
            d.noalias() += (im.weight * Scalar(2) * a * a * b) * consequenceDu(im, x, u);
        }
        return d;
    }
    /// Signed 2018 residual of implication i: [-g]_+ * s c.
    Scalar residual2018(int i, const VectorXs& x, const VectorXs& u) const {
        const auto& im = implication(i);
        return posPart(-im.trigger(x)) * consequenceVal(im, x, u);
    }
    /// Whether implication i is triggered at x (g < 0).
    bool triggered(int i, const VectorXs& x) const { return implication(i).trigger(x) < Scalar(0); }

    /// s_i c_i and its gradients (finite differences only on opt-in).
    Scalar consequenceVal(const Implication& im, const VectorXs& x, const VectorXs& u) const {
        return im.scale * im.consequence(x, u);
    }
    VectorXs triggerDx(const Implication& im, const VectorXs& x) const {
        if (im.trigger_dx) return im.trigger_dx(x);
        requireFd(im, "trigger_dx");
        VectorXs d(x.size());
        VectorXs xp = x, xm = x;
        for (int j = 0; j < x.size(); ++j) {
            const Scalar h = fdStep(x(j));
            xp(j) = x(j) + h; xm(j) = x(j) - h;
            d(j) = (im.trigger(xp) - im.trigger(xm)) / (Scalar(2) * h);
            xp(j) = x(j); xm(j) = x(j);
        }
        return d;
    }
    VectorXs consequenceDx(const Implication& im, const VectorXs& x, const VectorXs& u) const {
        if (im.consequence_dx) return (im.scale * im.consequence_dx(x, u)).eval();
        requireFd(im, "consequence_dx");
        VectorXs d(x.size());
        VectorXs xp = x, xm = x;
        for (int j = 0; j < x.size(); ++j) {
            const Scalar h = fdStep(x(j));
            xp(j) = x(j) + h; xm(j) = x(j) - h;
            d(j) = im.scale * (im.consequence(xp, u) - im.consequence(xm, u)) / (Scalar(2) * h);
            xp(j) = x(j); xm(j) = x(j);
        }
        return d;
    }
    VectorXs consequenceDu(const Implication& im, const VectorXs& x, const VectorXs& u) const {
        if (im.consequence_du) return (im.scale * im.consequence_du(x, u)).eval();
        requireFd(im, "consequence_du");
        VectorXs d(u.size());
        VectorXs up = u, um = u;
        for (int j = 0; j < u.size(); ++j) {
            const Scalar h = fdStep(u(j));
            up(j) = u(j) + h; um(j) = u(j) - h;
            d(j) = im.scale * (im.consequence(x, up) - im.consequence(x, um)) / (Scalar(2) * h);
            up(j) = u(j); um(j) = u(j);
        }
        return d;
    }

    static Scalar posPart(Scalar v) { return v > Scalar(0) ? v : Scalar(0); }

private:
    Implication& last() {
        if (impls_.empty()) throw std::logic_error("ctcs_stc::Spec: call add() before configuring an implication");
        return impls_.back();
    }
    static void requireFd(const Implication& im, const char* what) {
        if (!im.allow_fd)
            throw std::logic_error("ctcs_stc::Spec: implication \"" + im.name + "\" has no " + what +
                                   ". Supply it, or call .finiteDifference() on that implication to "
                                   "accept central differences deliberately.");
    }
    Scalar fdStep(Scalar v) const {
        const Scalar av = v > Scalar(0) ? v : -v;
        return Scalar(fd_h_) * (Scalar(1) + av);
    }

    Form form_;
    Scalar eps_;
    double fd_h_ = 1e-6;
    std::vector<Implication> impls_;
};

using Spec = SpecTpl<double>;
using Implication = ImplicationTpl<double>;

}  // namespace ctcs_stc
