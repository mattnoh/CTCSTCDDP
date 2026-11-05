/// @file spec.hpp
/// @brief State-triggered (STC) and continuous-time (CTCS) constraint rows, solver-agnostic.
/// @details Depends on Eigen only. `stage_function.hpp` exposes the rows to Aligator.
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
/// Three emitters from ONE stored specification (Form):
///
///   Original2018  one row per implication at the node:  [-g_i]_+ * c_i <= 0
///   Density       one row per interval:   I_k - eps <= 0
///   DensitySplit  one row per implication per interval:  I_{k,i} - eps_i <= 0
///
///   I_k = integral over [t_k, t_k+1] of  H(x(t),u(t)),  H = sum_i w_i [-g_i]_+^2 [s_i c_i]_+^2
///
/// I_k depends on (x_k, u_k) only: it is evaluated along the host's continuous
/// dynamics from x_k (RK4 stage points, Simpson weights), so no accumulator state
/// is needed.  This is the non-augmented form of Elango et al. 2025 (eq. 34), valid
/// for nonlinear dynamics too, and numerically identical to RK4 on y' = H.
///
/// SCALING.  The Density form adds every implication into one row against one eps,
/// so write each consequence as a DIMENSIONLESS FRACTION OF ITS OWN LIMIT
/// (`scale(1/limit)`, or divide inside the lambda) and keep weights at 1 unless you
/// want a deliberate priority:
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
/// @author <AUTHOR>

#pragma once

#include <Eigen/Dense>

#include <array>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctcs_stc {

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/// Which published transcription emits the rows.
enum class Form {
    Original2018,  ///< nodal: [-g]_+ * c <= 0, one row per implication (Szmuk 2018)
    Density,       ///< interval: integral of [-g]_+^2 [c]_+^2 over the step - eps <= 0 (CT-cSTC)
    DensitySplit   ///< interval, ONE ROW PER IMPLICATION: I_i - eps_i <= 0 (elementwise, Elango/OpenSCvx)
};

/// Continuous dynamics the Density form integrates against.
///
/// These are the HOST's continuous rates and their Jacobians, not the discrete
/// map.  `tau` is the offset within the interval, so a host with time-varying
/// externals (a target's predicted acceleration, say) can sample them at the
/// stage times rather than freezing them at the node.
template <typename Scalar>
struct IntervalDynamics {
    std::function<Vector<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&, Scalar)> xdot;
    std::function<Matrix<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&, Scalar)> fx;
    std::function<Matrix<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&, Scalar)> fu;

    /// Index of the time-dilation control, or -1 for a fixed step.
    int idx_theta = -1;
    /// Step used when idx_theta < 0.
    double dt = 0.0;

    /// Number of RK4 steps the interval is integrated with.  1 is one RK4 step
    /// over the whole interval (4 stage points).  M > 1 integrates M equal
    /// substeps of Th/M, exactly as a host whose discrete map is M chained RK4
    /// steps does, so the constraint quadrature matches that host's own.
    int substeps = 1;

    /// Optional: the control the DENSITY sees at a stage point, as a function
    /// of that stage point x, the stage control u, and the relative position
    /// beta in [0,1] within the interval.  Needed when a consequence reads the
    /// control and the host does not hold it constant across the interval -- a
    /// first-order hold with the left knot in the state, for instance, gives
    /// control(x,u,beta) = (1-beta) x_left + beta u_right.  Unset means the
    /// density sees the stage control u unchanged at every stage point.
    /// beta takes the RK4 stage positions (j + {0, 1/2, 1/2, 1}) / substeps.
    std::function<Vector<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&, Scalar)> control;
    /// d control / d x at the stage point, and d control / d u.  Required with
    /// `control`.  beta does not depend on Th, so there is no Th derivative.
    std::function<Matrix<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&, Scalar)> control_dx;
    std::function<Matrix<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&, Scalar)> control_du;
};

/// One implication `g(x) < 0 ==> c(x,u) <= 0` with its own weight.
///
/// Only `trigger` and `consequence` are required.  Jacobians default to central
/// finite differences; supply the analytic versions when the cost matters or
/// when the primitive has a kink you want handled deliberately.
template <typename Scalar>
struct Implication {
    std::string name;

    /// g(x): fires where NEGATIVE.
    std::function<Scalar(const Vector<Scalar>&)> trigger;
    /// c(x,u): satisfied where <= 0.
    std::function<Scalar(const Vector<Scalar>&, const Vector<Scalar>&)> consequence;

    /// Optional analytic Jacobians (row vectors).  Empty => finite difference.
    std::function<Vector<Scalar>(const Vector<Scalar>&)> trigger_dx;
    std::function<Vector<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)> consequence_dx;
    std::function<Vector<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)> consequence_du;

    /// Constraint SCALE s_i: the consequence is used as s_i * c_i everywhere (value and
    /// both Jacobians), so the row sees a rescaled constraint with the SAME zero set.
    /// This is the papers' "scale the constraint functions to similar orders of
    /// magnitude" knob -- set it to 1/limit to make the row a dimensionless fraction of
    /// its own limit.  Distinct from `weight`: scale changes the constraint the row
    /// enforces, weight only reweights it in the sum (and w [c]_+^2 == [sqrt(w) c]_+^2,
    /// so a scale s is the same as a weight s^2 -- use whichever states your intent).
    Scalar scale = Scalar(1);

    /// Density weight w_i.  Unused by Original2018, which has no aggregation.
    Scalar weight = Scalar(1);

    /// Per-implication interval budget, used by Form::DensitySplit only.
    /// Negative means "inherit the constraint-wide eps".
    Scalar epsilon = Scalar(-1);

    /// Opt in to central differences for THIS implication's missing Jacobians.
    /// Off by default: a silent FD fallback is ~16% slower per iteration and
    /// noisy enough to move a long solve into a different basin, and neither
    /// shows up as an error.  Ask for it deliberately or supply the Jacobian.
    bool allow_fd = false;
};

template <typename Scalar>
class Spec {
public:
    /// @param form  which transcription emits the rows
    /// @param eps   interval budget (the CTCS epsilon-relaxation); ignored by Original2018
    explicit Spec(Form form = Form::Density, double eps = 0.0)
        : form_(form), eps_(Scalar(eps)) {
    }

    // ── Authoring ────────────────────────────────────────────────────────────
    // Chainable, so an implication reads as one statement:
    //
    //   stc.add("landing_speed")
    //      .triggerWhenNegative([](const auto& x){ return x(2) - ALT_TRIG; })
    //      .consequence([](const auto& x, const auto&){ return x.segment(3,3).norm() - V_MAX; })
    //      .weight(100.0);

    Spec& add(std::string name) {
        clearCache();
        Implication<Scalar> impl;
        impl.name = std::move(name);
        impls_.push_back(std::move(impl));
        refreshDim();
        return *this;
    }

    Spec& triggerWhenNegative(
        std::function<Scalar(const Vector<Scalar>&)> g,
        std::function<Vector<Scalar>(const Vector<Scalar>&)> dg_dx = {}) {
        last().trigger = std::move(g);
        last().trigger_dx = std::move(dg_dx);
        clearCache();
        return *this;
    }

    Spec& consequence(
        std::function<Scalar(const Vector<Scalar>&, const Vector<Scalar>&)> c,
        std::function<Vector<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)> dc_dx = {},
        std::function<Vector<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)> dc_du = {}) {
        last().consequence = std::move(c);
        last().consequence_dx = std::move(dc_dx);
        last().consequence_du = std::move(dc_du);
        clearCache();
        return *this;
    }

    /// Density weight w_i.  See "SCALING" in the file header before setting this: the
    /// published form of the fold is an UNWEIGHTED sum and the rows are made
    /// commensurate by scaling each consequence instead.  w_i [c_i]_+^2 is identical to
    /// [sqrt(w_i) c_i]_+^2, so a weight is only ever a scaling written in the other
    /// place -- which is fine, but say which scale it encodes.
    /// Scale THIS implication's consequence (value and Jacobians) by s.  Pass 1/limit
    /// for the papers' normalisation; the zero set is unchanged.  See "SCALING" above.
    Spec& scale(Scalar s) { last().scale = s; clearCache(); return *this; }

    Spec& weight(Scalar w) { last().weight = w; clearCache(); return *this; }

    /// Per-implication interval budget for Form::DensitySplit.  Omit to
    /// inherit the constraint-wide eps.
    Spec& epsilon(Scalar e) { last().epsilon = e; clearCache(); return *this; }

    /// Accept central differences for whichever Jacobians this implication did
    /// not supply.  Use it for genuinely awkward primitives (a quaternion
    /// boresight cone, say), not as a default.
    Spec& finiteDifference() { last().allow_fd = true; clearCache(); return *this; }

    // NO AD HELPERS -- see the note at the top of this file.  Supply analytic
    // Jacobians for the hot implications; FD is the safe default.

    // ── Configuration ────────────────────────────────────────────────────────
    void setForm(Form form) { form_ = form; refreshDim(); clearCache(); }
    /// Required by Form::Density; unused by Original2018.
    void setIntervalDynamics(IntervalDynamics<Scalar> dyn) { dyn_ = std::move(dyn); clearCache(); }
    Form getForm() const { return form_; }
    void setEpsilon(double eps) { eps_ = Scalar(eps); }
    Scalar getEpsilon() const { return eps_; }
    /// Number of rows c() returns: 1 for Density, one per implication otherwise.
    int dim() const { return dim_; }
    void setFdStep(double h) { fd_h_ = h; clearCache(); }
    int numImplications() const { return static_cast<int>(impls_.size()); }
    const Implication<Scalar>& implication(int i) const { return impls_[i]; }

    // ── Pointwise kernel (host-facing, form-independent) ─────────────────────

    /// sum_i w_i [-g_i]_+^2 [c_i]_+^2 -- the CT-cSTC integrand, WITHOUT eps.
    /// This is what an accumulator or interval host integrates over a step.
    Scalar density(const Vector<Scalar>& x, const Vector<Scalar>& u, int imp = -1) const {
        Scalar h = Scalar(0);
        for (size_t i = 0; i < impls_.size(); ++i) {
            if (imp >= 0 && static_cast<int>(i) != imp) continue;
            const auto& im = impls_[i];
            const Scalar a = posPart(-im.trigger(x));
            const Scalar b = posPart(consequenceVal(im, x, u));
            h += im.weight * a * a * b * b;
        }
        return h;
    }

    /// Interval integral of the density by the RK4 stage points with Simpson
    /// weights (1,2,2,1)/6, over `substeps` equal RK4 steps of the host's
    /// continuous dynamics -- exactly what RK4 on an accumulator y' = H gives.
    Scalar intervalIntegral(const Vector<Scalar>& x, const Vector<Scalar>& u, int imp = -1) const {
        return valueIntegral(x, u, imp);
    }

    /// Budget of implication i for DensitySplit: its own eps, else the shared one.
    Scalar epsilonOf(int i) const {
        return impls_[static_cast<size_t>(i)].epsilon >= Scalar(0)
                   ? impls_[static_cast<size_t>(i)].epsilon : eps_;
    }

    /// Signed 2018 residual of one implication: [-g]_+ * c.
    Scalar residual2018(int i, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return posPart(-impls_[i].trigger(x)) * consequenceVal(impls_[i], x, u);
    }

    /// Whether implication i is triggered at x (g < 0).  Diagnostics/auditing.
    bool triggered(int i, const Vector<Scalar>& x) const {
        return impls_[i].trigger(x) < Scalar(0);
    }

    // ── Rows c(x,u) <= 0 and their derivatives ───────────────────────────────

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if (form_ == Form::Original2018) {
            Vector<Scalar> out(impls_.size());
            for (size_t i = 0; i < impls_.size(); ++i)
                out(static_cast<int>(i)) = residual2018(static_cast<int>(i), x, u);
            return out;
        }
        if (form_ == Form::DensitySplit) {
            const auto& g = cachedSplit(x, u);
            Vector<Scalar> out(impls_.size());
            for (size_t i = 0; i < impls_.size(); ++i) {
                const int ii = static_cast<int>(i);
                out(ii) = g[i].value - epsilonOf(ii);
            }
            return out;
        }
        Vector<Scalar> out(1);
        out(0) = cachedValue(x, u) - eps_;
        return out;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if (form_ == Form::Density) {
            Matrix<Scalar> J = Matrix<Scalar>::Zero(1, x.size());
            J.row(0) = cachedAggregate(x, u).dx.transpose();
            return J;
        }
        if (form_ == Form::DensitySplit) {
            Matrix<Scalar> J = Matrix<Scalar>::Zero(dim_, x.size());
            const auto& g = cachedSplit(x, u);
            for (size_t i = 0; i < impls_.size(); ++i)
                J.row(static_cast<int>(i)) = g[i].dx.transpose();
            return J;
        }
        Matrix<Scalar> J = Matrix<Scalar>::Zero(dim_, x.size());
        for (size_t i = 0; i < impls_.size(); ++i) {
            const auto& im = impls_[i];
            const Scalar g = im.trigger(x);
            const Scalar cv = consequenceVal(im, x, u);
            const Vector<Scalar> dg = triggerDx(im, x);
            const Vector<Scalar> dc = consequenceDx(im, x, u);
            const int row = (form_ == Form::Original2018) ? static_cast<int>(i) : 0;

            if (form_ == Form::Original2018) {
                // r = a*c with a = [-g]_+;  da/dx = -dg/dx where g < 0, else 0.
                const Scalar a = posPart(-g);
                if (g < Scalar(0)) J.row(row).noalias() += (-cv) * dg.transpose();
                J.row(row).noalias() += a * dc.transpose();
            } else {
                // H = w a^2 b^2,  a = [-g]_+, b = [c]_+
                const Scalar a = posPart(-g), b = posPart(cv);
                if (a > Scalar(0) && b > Scalar(0)) {
                    J.row(row).noalias() += (im.weight * Scalar(-2) * a * b * b) * dg.transpose();
                    J.row(row).noalias() += (im.weight * Scalar(2) * a * a * b) * dc.transpose();
                }
            }
        }
        return J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if (form_ == Form::Density) {
            Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
            J.row(0) = cachedAggregate(x, u).du.transpose();
            return J;
        }
        if (form_ == Form::DensitySplit) {
            Matrix<Scalar> J = Matrix<Scalar>::Zero(dim_, u.size());
            const auto& g = cachedSplit(x, u);
            for (size_t i = 0; i < impls_.size(); ++i)
                J.row(static_cast<int>(i)) = g[i].du.transpose();
            return J;
        }
        Matrix<Scalar> J = Matrix<Scalar>::Zero(dim_, u.size());
        for (size_t i = 0; i < impls_.size(); ++i) {
            const auto& im = impls_[i];
            const Scalar g = im.trigger(x);           // trigger is state-only: no du term
            const Scalar cv = consequenceVal(im, x, u);
            const Vector<Scalar> dc = consequenceDu(im, x, u);
            const int row = (form_ == Form::Original2018) ? static_cast<int>(i) : 0;
            const Scalar a = posPart(-g);

            if (form_ == Form::Original2018) {
                J.row(row).noalias() += a * dc.transpose();
            } else {
                const Scalar b = posPart(cv);
                if (a > Scalar(0) && b > Scalar(0))
                    J.row(row).noalias() += (im.weight * Scalar(2) * a * a * b) * dc.transpose();
            }
        }
        return J;
    }


    // For I=sum r_j^2, retain 2*sum J_rj' J_rj in the constraint
    // Lagrangian. Omitting it leaves only the slack-elimination curvature,
    // which is not the curvature of the nonlinear CTCS residuals.
    Matrix<Scalar> lagrangianHessian(const Vector<Scalar>& x,
        const Vector<Scalar>& u,const Vector<Scalar>& multiplier) const {
        const int nx=x.size(),nu=u.size();
        Matrix<Scalar> hessian=Matrix<Scalar>::Zero(nx+nu,nx+nu);
        if(form_==Form::Original2018)return hessian;
        const auto st=stages(x,u);
        if(!(st.Th>Scalar(0)))throw std::invalid_argument("CTCS curvature requires positive duration");
        for(const auto& sub:st.sub)for(const auto& pt:sub){
            const auto& uc=pointControl(pt,u);
            for(size_t i=0;i<impls_.size();++i){
                const auto& im=impls_[i];
                const Scalar a=posPart(-im.trigger(pt.p)),b=posPart(consequenceVal(im,pt.p,uc));
                if(!(a>Scalar(0)&&b>Scalar(0)))continue;
                const Vector<Scalar> gp=-b*triggerDx(im,pt.p)+a*consequenceDx(im,pt.p,uc);
                const Vector<Scalar> gu=a*consequenceDu(im,pt.p,uc);
                Vector<Scalar> j(nx+nu);
                j.head(nx)=pt.Sx.transpose()*gp;
                j.tail(nu)=pt.Su.transpose()*gp;
                if(pt.hooked){j.head(nx)+=pt.Cx.transpose()*gu;j.tail(nu)+=pt.Cu.transpose()*gu;}
                else j.tail(nu)+=gu;
                if(dyn_.idx_theta>=0)j(nx+dyn_.idx_theta)+=gp.dot(pt.ST)
                    +(pt.hooked?gu.dot(pt.CT):Scalar(0))+a*b/(Scalar(2)*st.Th);
                const Scalar y=multiplier(form_==Form::DensitySplit?static_cast<int>(i):0);
                hessian.noalias()+=(Scalar(2)*y*st.h*pt.w*im.weight)*(j*j.transpose());
            }
        }
        return hessian;
    }

private:
    static Scalar posPart(Scalar v) { return v > Scalar(0) ? v : Scalar(0); }

    Implication<Scalar>& last() {
        // add() is always called first; this is a programming error otherwise.
        return impls_.back();
    }

    void refreshDim() {
        dim_ = (form_ == Form::Original2018 || form_ == Form::DensitySplit)
                          ? static_cast<int>(impls_.size())
                          : (impls_.empty() ? 0 : 1);
    }

    // ── CT-cSTC interval quadrature ──────────────────────────────────────────
    /// One RK4 stage point, with its sensitivities to the INTERVAL's inputs
    /// (the state x and control u at the start of the interval, and Th).
    struct Point {
        Vector<Scalar> p;       ///< stage point
        Matrix<Scalar> Sx;      ///< d p / d x
        Matrix<Scalar> Su;      ///< d p / d u
        Vector<Scalar> ST;      ///< d p / d Th
        Scalar w;               ///< Simpson weight, (1,2,2,1)/6
        bool hooked = false;    ///< a control hook is set
        Vector<Scalar> c;       ///< control the density sees (hooked only)
        Matrix<Scalar> Cx;      ///< d c / d x                  (hooked only)
        Matrix<Scalar> Cu;      ///< d c / d u                  (hooked only)
        Vector<Scalar> CT;      ///< d c / d Th                 (hooked only)
    };
    struct Stages {
        Scalar Th;
        Scalar h;                                   ///< Th / substeps
        int M;
        std::vector<std::array<Point, 4>> sub;      ///< per substep
    };

    const Vector<Scalar>& pointControl(const Point& pt, const Vector<Scalar>& u) const {
        return pt.hooked ? pt.c : u;
    }

    void requireIntervalDynamics() const {
        if (!dyn_.xdot || !dyn_.fx || !dyn_.fu) {
            throw std::logic_error(
                "ctcs_stc::Spec: Form::Density integrates over the interval and "
                "needs the host's continuous dynamics -- call setIntervalDynamics() with "
                "xdot/fx/fu, and either idx_theta (free time) or dt (fixed step).");
        }
        if (dyn_.idx_theta < 0 && !(dyn_.dt > 0.0)) {
            throw std::logic_error(
                "ctcs_stc::Spec: set IntervalDynamics::idx_theta for free time, "
                "or a positive dt for a fixed step.");
        }
        if (dyn_.substeps < 1) {
            throw std::logic_error("ctcs_stc::Spec: IntervalDynamics::substeps must be >= 1.");
        }
        if (dyn_.control && (!dyn_.control_dx || !dyn_.control_du)) {
            throw std::logic_error(
                "ctcs_stc::Spec: IntervalDynamics::control needs control_dx and control_du.");
        }
    }

    /// Value-only interval integral: the same RK4 stage points, Simpson weights
    /// and control hook as stages(), WITHOUT the Sx/Su/ST sensitivity chain, so
    /// a constraint value (a line-search trial, say) costs only the integration.
    Scalar valueIntegral(const Vector<Scalar>& x, const Vector<Scalar>& u, int imp) const {
        requireIntervalDynamics();
        const int M = dyn_.substeps;
        const Scalar Th = (dyn_.idx_theta >= 0) ? u(dyn_.idx_theta) : Scalar(dyn_.dt);
        const Scalar h = Th / Scalar(M), h2 = Scalar(0.5) * h;
        static const Scalar kW[4] = {Scalar(1) / Scalar(6), Scalar(2) / Scalar(6),
                                     Scalar(2) / Scalar(6), Scalar(1) / Scalar(6)};
        static const Scalar kC[4] = {Scalar(0), Scalar(0.5), Scalar(0.5), Scalar(1)};
        const bool hooked = static_cast<bool>(dyn_.control);
        Vector<Scalar> xj = x;
        Scalar total = Scalar(0);
        for (int j = 0; j < M; ++j) {
            const Scalar t0 = Scalar(j) * h;
            const Vector<Scalar> k1 = dyn_.xdot(xj, u, t0);
            const Vector<Scalar> x2 = xj + h2 * k1;
            const Vector<Scalar> k2 = dyn_.xdot(x2, u, t0 + h2);
            const Vector<Scalar> x3 = xj + h2 * k2;
            const Vector<Scalar> k3 = dyn_.xdot(x3, u, t0 + h2);
            const Vector<Scalar> x4 = xj + h * k3;
            const Vector<Scalar> P[4] = {xj, x2, x3, x4};
            Scalar sum = Scalar(0);
            for (int i = 0; i < 4; ++i) {
                if (hooked) {
                    const Vector<Scalar> pu = dyn_.control(P[i], u, (Scalar(j) + kC[i]) / Scalar(M));
                    sum += kW[i] * density(P[i], pu, imp);
                } else {
                    sum += kW[i] * density(P[i], u, imp);
                }
            }
            total += h * sum;
            if (j + 1 < M) {
                const Vector<Scalar> k4 = dyn_.xdot(x4, u, t0 + h);
                xj = xj + (h / Scalar(6)) * (k1 + Scalar(2) * k2 + Scalar(2) * k3 + k4);
            }
        }
        return total;
    }

    Stages stages(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        requireIntervalDynamics();
        const bool hooked = static_cast<bool>(dyn_.control);
        const int nx = static_cast<int>(x.size());
        const int nu = static_cast<int>(u.size());
        const int M = dyn_.substeps;

        Stages s;
        s.Th = (dyn_.idx_theta >= 0) ? u(dyn_.idx_theta) : Scalar(dyn_.dt);
        s.M = M;
        s.h = s.Th / Scalar(M);
        const Scalar h = s.h, h2 = Scalar(0.5) * h;
        const Matrix<Scalar> I = Matrix<Scalar>::Identity(nx, nx);
        static const Scalar kW[4] = {Scalar(1) / Scalar(6), Scalar(2) / Scalar(6),
                                     Scalar(2) / Scalar(6), Scalar(1) / Scalar(6)};
        static const Scalar kC[4] = {Scalar(0), Scalar(0.5), Scalar(0.5), Scalar(1)};

        // State at the start of substep j, and its sensitivity to the interval
        // inputs.  j == 0 is the identity chain, which is skipped outright so
        // the single-substep case performs exactly the arithmetic it always has.
        Vector<Scalar> xj = x;
        Matrix<Scalar> Phi, Psi;          // d xj / d x,  d xj / d u
        Vector<Scalar> psiT;              // d xj / d Th
        s.sub.resize(static_cast<size_t>(M));

        for (int j = 0; j < M; ++j) {
            const Scalar t0 = Scalar(j) * h;
            const Vector<Scalar> k1 = dyn_.xdot(xj, u, t0);
            const Vector<Scalar> x2 = xj + h2 * k1;
            const Vector<Scalar> k2 = dyn_.xdot(x2, u, t0 + h2);
            const Vector<Scalar> x3 = xj + h2 * k2;
            const Vector<Scalar> k3 = dyn_.xdot(x3, u, t0 + h2);
            const Vector<Scalar> x4 = xj + h * k3;

            const Matrix<Scalar> A1 = dyn_.fx(xj, u, t0),      B1 = dyn_.fu(xj, u, t0);
            const Matrix<Scalar> A2 = dyn_.fx(x2, u, t0 + h2), B2 = dyn_.fu(x2, u, t0 + h2);
            const Matrix<Scalar> A3 = dyn_.fx(x3, u, t0 + h2), B3 = dyn_.fu(x3, u, t0 + h2);

            // local sensitivities, w.r.t. the substep's own start state and h
            const Matrix<Scalar> d1x = A1, d1u = B1;
            const Matrix<Scalar> d2x = A2 * (I + h2 * d1x);
            const Matrix<Scalar> d2u = A2 * (h2 * d1u) + B2;
            const Vector<Scalar> d2T = A2 * (Scalar(0.5) * k1);
            const Matrix<Scalar> d3x = A3 * (I + h2 * d2x);
            const Matrix<Scalar> d3u = A3 * (h2 * d2u) + B3;
            const Vector<Scalar> d3T = A3 * (Scalar(0.5) * k2 + h2 * d2T);

            Vector<Scalar> P[4] = {xj, x2, x3, x4};
            Matrix<Scalar> LSx[4] = {I, I + h2 * d1x, I + h2 * d2x, I + h * d3x};
            Matrix<Scalar> LSu[4] = {Matrix<Scalar>::Zero(nx, nu), h2 * d1u, h2 * d2u, h * d3u};
            Vector<Scalar> LST[4] = {Vector<Scalar>::Zero(nx), Scalar(0.5) * k1,
                                     Scalar(0.5) * k2 + h2 * d2T, k3 + h * d3T};

            auto& pts = s.sub[static_cast<size_t>(j)];
            for (int i = 0; i < 4; ++i) {
                Point& pt = pts[i];
                pt.p = P[i];
                pt.w = kW[i];
                if (j == 0) {
                    pt.Sx = LSx[i];
                    pt.Su = LSu[i];
                    pt.ST = (M == 1) ? LST[i] : Vector<Scalar>(LST[i] / Scalar(M));
                } else {
                    pt.Sx = LSx[i] * Phi;
                    pt.Su = LSx[i] * Psi + LSu[i];
                    pt.ST = LSx[i] * psiT + LST[i] / Scalar(M);
                }
                if (hooked) {
                    const Scalar beta = (Scalar(j) + kC[i]) / Scalar(M);
                    pt.hooked = true;
                    pt.c = dyn_.control(pt.p, u, beta);
                    const Matrix<Scalar> Cp = dyn_.control_dx(pt.p, u, beta);
                    pt.Cx = Cp * pt.Sx;
                    pt.Cu = Cp * pt.Su + dyn_.control_du(pt.p, u, beta);
                    pt.CT = Cp * pt.ST;
                }
            }

            if (j + 1 < M) {
                // Advance to the next substep: the full RK4 step and its chain.
                const Vector<Scalar> k4 = dyn_.xdot(x4, u, t0 + h);
                const Matrix<Scalar> A4 = dyn_.fx(x4, u, t0 + h), B4 = dyn_.fu(x4, u, t0 + h);
                const Matrix<Scalar> d4x = A4 * LSx[3];
                const Matrix<Scalar> d4u = A4 * LSu[3] + B4;
                const Vector<Scalar> d4T = A4 * LST[3];
                const Matrix<Scalar> Jx = I + (h / Scalar(6)) * (d1x + Scalar(2) * d2x + Scalar(2) * d3x + d4x);
                const Matrix<Scalar> Ju = (h / Scalar(6)) * (d1u + Scalar(2) * d2u + Scalar(2) * d3u + d4u);
                const Vector<Scalar> Jh = (k1 + Scalar(2) * k2 + Scalar(2) * k3 + k4) / Scalar(6)
                                        + (h / Scalar(6)) * (Scalar(2) * d2T + Scalar(2) * d3T + d4T);
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
        }
        return s;
    }

    struct IntervalGrad { Scalar value = Scalar(0); Vector<Scalar> dx, du; };

    /// Accumulate one implication's (or the aggregate's) interval gradient.
    ///   dI/dx  = h sum_j sum_i w_i (Sx_i^T Hx_i + Cx_i^T Hc_i)
    ///   dI/du  = h sum_j sum_i w_i (Su_i^T Hx_i + Cu_i^T Hc_i)   <- direct term
    ///   dI/dTh = sum_j [ (1/M) sum_i w_i H_i + h sum_i w_i (Hx_i.ST_i + Hc_i.CT_i) ]
    /// Without a control hook Cx = 0, Cu = I and CT = 0, which is the original
    /// single-step formula term for term.
    IntervalGrad accumulate(const Stages& st, const Vector<Scalar>& x,
                            const Vector<Scalar>& u, int imp) const {
        IntervalGrad out;
        out.dx = Vector<Scalar>::Zero(x.size());
        out.du = Vector<Scalar>::Zero(u.size());
        Scalar dT = Scalar(0);
        for (const auto& sub : st.sub) {
            Vector<Scalar> sdx = Vector<Scalar>::Zero(x.size());
            Vector<Scalar> sdu = Vector<Scalar>::Zero(u.size());
            Scalar sum_h = Scalar(0), gT = Scalar(0);
            for (int i = 0; i < 4; ++i) {
                const Point& pt = sub[i];
                const Vector<Scalar>& uc = pointControl(pt, u);
                const Vector<Scalar> Hx = densityDx(pt.p, uc, imp);
                const Vector<Scalar> Hc = densityDu(pt.p, uc, imp);
                sum_h += pt.w * density(pt.p, uc, imp);
                if (pt.hooked) {
                    sdx.noalias() += pt.w * (pt.Sx.transpose() * Hx + pt.Cx.transpose() * Hc);
                    sdu.noalias() += pt.w * (pt.Su.transpose() * Hx + pt.Cu.transpose() * Hc);
                    gT += pt.w * (Hx.dot(pt.ST) + Hc.dot(pt.CT));
                } else {
                    sdx.noalias() += pt.w * (pt.Sx.transpose() * Hx);
                    sdu.noalias() += pt.w * (pt.Su.transpose() * Hx + Hc);
                    gT += pt.w * Hx.dot(pt.ST);
                }
            }
            sdx *= st.h;
            sdu *= st.h;
            out.dx += sdx;
            out.du += sdu;
            out.value += st.h * sum_h;
            dT += ((st.M == 1) ? sum_h : sum_h / Scalar(st.M)) + st.h * gT;
        }
        if (dyn_.idx_theta >= 0) out.du(dyn_.idx_theta) = dT;
        return out;
    }

    // ── evaluation cache ─────────────────────────────────────────────────────
    // The solver asks for c, cx and cu at the same (x,u) in turn.  Each of them
    // needs the whole interval integration, so the last point's result is kept
    // and reused instead of integrating three times.  Keyed on exact equality of
    // (x,u); every setter that changes what the row computes clears it.  Not
    // thread-safe, like the rest of a stage constraint's mutable state.
    struct CacheKey {
        bool valid = false;
        Vector<Scalar> x, u;
        bool matches(const Vector<Scalar>& xx, const Vector<Scalar>& uu) const {
            return valid && x.size() == xx.size() && u.size() == uu.size() && x == xx && u == uu;
        }
        void set(const Vector<Scalar>& xx, const Vector<Scalar>& uu) { x = xx; u = uu; valid = true; }
    };
    mutable CacheKey agg_key_, split_key_;
    mutable IntervalGrad agg_;
    mutable CacheKey val_key_;
    mutable Scalar val_ = Scalar(0);
    mutable std::vector<IntervalGrad> split_;
    void clearCache() const { agg_key_.valid = false; split_key_.valid = false; val_key_.valid = false; }

    /// Constraint VALUE at (x,u): the derivative cache if it already holds this
    /// point, else a value-only integration cached on its own.  A trial point
    /// that is never differentiated therefore never pays for sensitivities.
    Scalar cachedValue(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if (agg_key_.matches(x, u)) return agg_.value;
        if (!val_key_.matches(x, u)) {
            val_ = valueIntegral(x, u, -1);
            val_key_.set(x, u);
        }
        return val_;
    }

    const IntervalGrad& cachedAggregate(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if (!agg_key_.matches(x, u)) {
            agg_ = intervalJac(x, u);
            agg_key_.set(x, u);
        }
        return agg_;
    }
    const std::vector<IntervalGrad>& cachedSplit(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if (!split_key_.matches(x, u)) {
            split_ = intervalJacSplit(x, u);
            split_key_.set(x, u);
        }
        return split_;
    }

    IntervalGrad intervalJac(const Vector<Scalar>& x, const Vector<Scalar>& u,
                             int imp = -1) const {
        return accumulate(stages(x, u), x, u, imp);
    }

    /// Per-implication interval gradients with ONE stages() evaluation.
    /// Calling intervalJac() per implication would rebuild the whole RK4
    /// sensitivity chain n times; the chain does not depend on the implication.
    std::vector<IntervalGrad> intervalJacSplit(const Vector<Scalar>& x,
                                               const Vector<Scalar>& u) const {
        const Stages st = stages(x, u);
        const int n = static_cast<int>(impls_.size());
        std::vector<IntervalGrad> out;
        out.reserve(static_cast<size_t>(n));
        for (int m = 0; m < n; ++m) out.push_back(accumulate(st, x, u, m));
        return out;
    }

    /// Gradient of the aggregate density w.r.t. x (and u), reusing the same
    /// per-implication chain rule the pointwise Density rows use.
    Vector<Scalar> densityDx(const Vector<Scalar>& x, const Vector<Scalar>& u,
                             int imp = -1) const {
        Vector<Scalar> d = Vector<Scalar>::Zero(x.size());
        for (size_t idx = 0; idx < impls_.size(); ++idx) {
            if (imp >= 0 && static_cast<int>(idx) != imp) continue;
            const auto& im = impls_[idx];
            const Scalar a = posPart(-im.trigger(x)), b = posPart(consequenceVal(im, x, u));
            if (a <= Scalar(0) || b <= Scalar(0)) continue;
            d.noalias() += (im.weight * Scalar(-2) * a * b * b) * triggerDx(im, x);
            d.noalias() += (im.weight * Scalar(2) * a * a * b) * consequenceDx(im, x, u);
        }
        return d;
    }

    Vector<Scalar> densityDu(const Vector<Scalar>& x, const Vector<Scalar>& u,
                             int imp = -1) const {
        Vector<Scalar> d = Vector<Scalar>::Zero(u.size());
        for (size_t idx = 0; idx < impls_.size(); ++idx) {
            if (imp >= 0 && static_cast<int>(idx) != imp) continue;
            const auto& im = impls_[idx];
            const Scalar a = posPart(-im.trigger(x)), b = posPart(consequenceVal(im, x, u));
            if (a <= Scalar(0) || b <= Scalar(0)) continue;
            d.noalias() += (im.weight * Scalar(2) * a * a * b) * consequenceDu(im, x, u);
        }
        return d;
    }

    Vector<Scalar> triggerDx(const Implication<Scalar>& im, const Vector<Scalar>& x) const {
        if (im.trigger_dx) return im.trigger_dx(x);
        if (!im.allow_fd) throw std::logic_error(
            "ctcs_stc::Spec: implication \"" + im.name + "\" has no trigger_dx. "
            "Supply it, or call .finiteDifference() on that implication to accept "
            "central differences deliberately.");
        Vector<Scalar> d = Vector<Scalar>::Zero(x.size());
        Vector<Scalar> xp = x, xm = x;
        for (int j = 0; j < x.size(); ++j) {
            const Scalar h = fdStep(x(j));
            xp(j) = x(j) + h; xm(j) = x(j) - h;
            d(j) = (im.trigger(xp) - im.trigger(xm)) / (Scalar(2) * h);
            xp(j) = x(j); xm(j) = x(j);
        }
        return d;
    }

    /// s_i * c_i(x,u) -- the only place the raw consequence is read for a value.
    Scalar consequenceVal(const Implication<Scalar>& im,
                          const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return im.scale * im.consequence(x, u);
    }

    Vector<Scalar> consequenceDx(const Implication<Scalar>& im,
                                 const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if (im.consequence_dx) return (im.scale * im.consequence_dx(x, u)).eval();
        if (!im.allow_fd) throw std::logic_error(
            "ctcs_stc::Spec: implication \"" + im.name + "\" has no consequence_dx. "
            "Supply it, or call .finiteDifference() on that implication to accept "
            "central differences deliberately.");
        Vector<Scalar> d = Vector<Scalar>::Zero(x.size());
        Vector<Scalar> xp = x, xm = x;
        for (int j = 0; j < x.size(); ++j) {
            const Scalar h = fdStep(x(j));
            xp(j) = x(j) + h; xm(j) = x(j) - h;
            d(j) = im.scale * (im.consequence(xp, u) - im.consequence(xm, u)) / (Scalar(2) * h);
            xp(j) = x(j); xm(j) = x(j);
        }
        return d;
    }

    Vector<Scalar> consequenceDu(const Implication<Scalar>& im,
                                 const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        if (im.consequence_du) return (im.scale * im.consequence_du(x, u)).eval();
        if (!im.allow_fd) throw std::logic_error(
            "ctcs_stc::Spec: implication \"" + im.name + "\" has no consequence_du. "
            "Supply it, or call .finiteDifference() on that implication to accept "
            "central differences deliberately.");
        Vector<Scalar> d = Vector<Scalar>::Zero(u.size());
        Vector<Scalar> up = u, um = u;
        for (int j = 0; j < u.size(); ++j) {
            const Scalar h = fdStep(u(j));
            up(j) = u(j) + h; um(j) = u(j) - h;
            d(j) = im.scale * (im.consequence(x, up) - im.consequence(x, um)) / (Scalar(2) * h);
            up(j) = u(j); um(j) = u(j);
        }
        return d;
    }

    Scalar fdStep(Scalar v) const {
        const Scalar av = v > Scalar(0) ? v : -v;
        return Scalar(fd_h_) * (Scalar(1) + av);
    }

    Form form_;
    Scalar eps_;
    IntervalDynamics<Scalar> dyn_;
    double fd_h_ = 1e-6;
    std::vector<Implication<Scalar>> impls_;
    int dim_ = 0;
};

}  // namespace ctcs_stc
