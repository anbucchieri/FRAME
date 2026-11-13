"""
Pipeline orchestration -- the single entry point of the metric engine.

`analyze(cap, cfg)` runs exactly the layers the data supports and returns a result dict plus a
`capability` report explaining, per layer, whether it ran and why.
"""
from __future__ import annotations
from . import task_space, joint_space, dynamics, pca, passivity


def capability(cap):
    """Which layers are computable from this capture, with a human-readable reason when not."""
    rep = {
        "task_space": {"ok": cap.has_ee,
                       "reason": "" if cap.has_ee else "needs end-effector positions"},
        "joint_space": {"ok": cap.n_dof > 0 and len(cap.reps) > 0,
                        "reason": "" if cap.n_dof else "needs joint angles"},
        "dynamics": {"ok": cap.has_torque,
                     "reason": "" if cap.has_torque else "needs torques or opt-in ID"},
    }
    rep["passivity"] = {"ok": cap.has_torque,
                         "reason": "" if cap.has_torque else "needs torques or opt-in ID",
                         "gravity": "" if cap.has_gravity else "no gravity torque provided"}
    rep["pca"] = {"ok": cap.n_dof >= 2 and len(cap.reps) > 0,
                  "reason": "" if (cap.n_dof >= 2 and cap.reps) else "needs >= 2 joint angles"}
    return rep


def analyze(cap, cfg, urdf_limits=None, urdf_vel_limits=None, layers=None):
    """Run the gated pipeline. `layers` optionally restricts to a subset (e.g. after a cheap
    re-weight only Layer 2 needs recompute)."""
    caps = capability(cap)
    want = set(layers) if layers else {"task_space", "joint_space", "dynamics", "passivity",
                                       "pca"}
    results = {"capability": caps}

    if "task_space" in want and caps["task_space"]["ok"]:
        results["task_space"] = task_space.compute(cap, cfg)
    if "joint_space" in want and caps["joint_space"]["ok"]:
        results["joint_space"] = joint_space.compute(cap, cfg, urdf_limits, urdf_vel_limits)
    if "dynamics" in want and caps["dynamics"]["ok"]:
        results["dynamics"] = dynamics.compute(cap, cfg, urdf_limits=urdf_limits)
    if "passivity" in want and caps["passivity"]["ok"]:
        results["passivity"] = passivity.compute(cap, cfg)
    if "pca" in want and caps["pca"]["ok"]:
        results["pca"] = pca.compute(cap, cfg)
    return results
