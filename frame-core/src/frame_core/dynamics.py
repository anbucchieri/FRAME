"""
Layer 3 -- dynamic (torque) requirements. Requires torques (gates on cap.has_torque).

Per-DOF, per-repetition:
  * peak / RMS torque [N.m]         -> actuator peak & continuous (thermal) rating.
  * peak power +/- [W]              -> actuator power rating and whether negative (regen/
                                       dissipation) power is present.
  * net cycle work |W| = |oint tau dq| [J]  -> energy per cycle -> battery / actuator energy.
  * quasi-stiffness k [N.m/rad] + R^2       -> passive-elastic feasibility (high R^2 & stable
                                       k => a spring/elastomer can do the job).
  * torque-speed operating cloud    -> demanded (tau, omega) points vs a candidate motor
                                       envelope (returned as raw samples for plotting), with a
                                       95% confidence ELLIPSE summarising the cloud's shape.
  * capacity ratios                 -> RMS and peak torque as a fraction of the joint's maximum
                                       torque capacity (the URDF `effort` limit).

_loop_work and _quasi_stiffness are the exact implementations from torque_analysis.py.
"""
from __future__ import annotations
import numpy as np
from .signal import derivative
from .types import summarize, limit_entry
from .anatomy import is_dynamics_validated, validation_note, dof_to_segment


# --- verbatim from torque_analysis.py (citable in the paper) --- #
def loop_work(q_rad, tau):
    """Net mechanical work over the closed cycle  W = oint tau dq  [J] (trapezoidal, loop
    closed back to the start).

    Note: it closes a trajectory that does not close by appending the first sample; read the
    `net_work` column with that in mind."""
    q = np.asarray(q_rad, float); t = np.asarray(tau, float)
    ok = np.isfinite(q) & np.isfinite(t)
    q, t = q[ok], t[ok]
    if q.size < 2:
        return float("nan")
    q = np.append(q, q[0]); t = np.append(t, t[0])
    return float(np.sum(0.5 * (t[:-1] + t[1:]) * np.diff(q)))


def quasi_stiffness(q_rad, tau):
    """Quasi-stiffness k = d tau / d q [N.m/rad] from tau ~ k q + b, with R^2. High R^2 and
    small |W| => linear-spring-like => a passive element of stiffness k fits."""
    q = np.asarray(q_rad, float); t = np.asarray(tau, float)
    ok = np.isfinite(q) & np.isfinite(t)
    q, t = q[ok], t[ok]
    if q.size < 3 or q.std() < 1e-6:
        return float("nan"), float("nan")
    k, b = np.polyfit(q, t, 1)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1.0 - np.sum((t - (k * q + b)) ** 2) / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(k), float(r2)


def _rep_dof_dynamics(q_rad, tau, fs):
    """All per-rep, per-DOF dynamic scalars in one pass."""
    q = np.asarray(q_rad, float)
    t = np.asarray(tau, float)
    w = derivative(q, fs)                       # rad/s
    power = t * w                               # W
    k, r2 = quasi_stiffness(q, t)
    return {
        "peak_torque": float(np.max(np.abs(t))),
        "rms_torque": float(np.sqrt(np.mean(t ** 2))),
        # power is undefined where the angle is masked (shoulder plane / rotation, gimbal lock)
        "peak_power_pos": float(np.nanmax(power)) if np.any(power > 0) else 0.0,
        "peak_power_neg": float(np.nanmin(power)) if np.any(power < 0) else 0.0,
        "net_work": loop_work(q, t),
        "k": k,
        "r2": r2,
    }


def confidence_ellipse(omega_dps, tau_Nm, n_std_chi2=5.991, n_points=101):
    """95% confidence ellipse of the (omega, tau) cloud, as a closed polyline.

    "An ellipsoid that models the relationship" on a 2D torque-speed plot is the covariance
    ellipse: eigen-decompose the 2x2 covariance, scale the semi-axes by sqrt(chi2_2dof * lambda),
    and 5.991 is the 95% quantile of chi-squared with 2 degrees of freedom.

    Returned as a SAMPLED POLYLINE so it can be drawn with the same primitive as the scatter. The
    parametric description rides along for anyone who wants the numbers.

    A degenerate cloud (constant torque, fewer than 3 points) returns None rather than a line.
    """
    w = np.asarray(omega_dps, float)
    t = np.asarray(tau_Nm, float)
    m = np.isfinite(w) & np.isfinite(t)
    w, t = w[m], t[m]
    if w.size < 3:
        return None
    pts = np.column_stack([w, t])
    centre = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)                  # ascending, orthonormal
    vals = np.clip(vals, 0.0, None)
    if vals[1] <= 0:                                  # no spread at all
        return None
    order = np.argsort(vals)[::-1]                    # major axis first
    vals, vecs = vals[order], vecs[:, order]
    semi = np.sqrt(n_std_chi2 * vals)
    th = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.column_stack([semi[0] * np.cos(th), semi[1] * np.sin(th)])
    poly = circle @ vecs.T + centre
    return {
        "omega_dps": poly[:, 0].tolist(),
        "tau_Nm": poly[:, 1].tolist(),
        "centre": [float(centre[0]), float(centre[1])],
        "semi_axes": [float(semi[0]), float(semi[1])],
        # An axis is a LINE, so the eigenvector's sign is arbitrary -- wrap into (-90, 90] or the
        # reported angle flips by 180 degrees on numerical noise alone.
        "angle_deg": float((np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])) + 90.0) % 180.0 - 90.0),
        "coverage": 0.95,
    }


def capacity_ratios(entry, effort_Nm):
    """RMS and peak torque as fractions of the joint's torque capacity.

    `effort_Nm` is the URDF effort limit, which in the DHM models carries a human joint torque
    capacity. Ratios use the same statistic the scalar column shows -- median RMS for the
    continuous demand, max peak for the worst case -- so the bar and the number agree.
    """
    if not effort_Nm or effort_Nm <= 0 or not np.isfinite(effort_Nm):
        return {}
    rms = entry["rms_torque"]["median"]
    peak = abs(entry["peak_torque"]["max"])
    return {
        "torque_capacity_Nm": float(effort_Nm),
        "rms_torque_ratio": float(rms / effort_Nm) if np.isfinite(rms) else float("nan"),
        "peak_torque_ratio": float(peak / effort_Nm) if np.isfinite(peak) else float("nan"),
    }


def compute(cap, cfg, return_cloud=True, urdf_limits=None):
    """Layer-3 result dict. Requires cap.has_torque.

    `urdf_limits` ({joint: {"effort": N.m, ...}}) enables the capacity ratios; without it the
    torque scalars are reported unnormalised, exactly as before.
    """
    keys = ("peak_torque", "rms_torque", "peak_power_pos", "peak_power_neg", "net_work", "k", "r2")
    acc = {name: {kk: [] for kk in keys} for name in cap.dof_names}
    cloud = {name: {"omega": [], "tau": []} for name in cap.dof_names}

    for r in cap.reps:
        for j, name in enumerate(cap.dof_names):
            d = _rep_dof_dynamics(r.angles[:, j], r.torques[:, j], cap.fs)
            for kk in keys:
                acc[name][kk].append(d[kk])
            if return_cloud:
                w = derivative(r.angles[:, j], cap.fs)
                cloud[name]["omega"].append(np.rad2deg(w))
                cloud[name]["tau"].append(r.torques[:, j])

    limits = urdf_limits or {}
    per_dof = {}
    for name in cap.dof_names:
        entry = {kk: summarize(acc[name][kk]) for kk in keys}
        if return_cloud:
            w_all = np.concatenate(cloud[name]["omega"])
            t_all = np.concatenate(cloud[name]["tau"])
            entry["torque_speed_cloud"] = {"omega_dps": w_all.tolist(), "tau_Nm": t_all.tolist()}
            entry["torque_speed_ellipse"] = confidence_ellipse(w_all, t_all)
        entry.update(capacity_ratios(entry, limit_entry(limits, name).get("effort")))
        entry["segment"] = dof_to_segment(name)      # where the N-pose map paints this load
        entry["dynamics_validated"] = is_dynamics_validated(name)
        entry["validation_note"] = validation_note(name)
        per_dof[name] = entry

    unvalidated = [n for n in cap.dof_names if not is_dynamics_validated(n)]
    return {"per_dof": per_dof,
            "validation": {"validated_scope": "upper limb (clavicular / shoulder / elbow / wrist)",
                           "unvalidated_dofs": unvalidated}}
