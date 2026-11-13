"""Gravity share and reversal metrics (frame_core.passivity): synthetic cases with known answers."""
import numpy as np
import pytest

from frame_core import passivity

FS = 240.0
G0 = 9.81


def test_pure_gravity_torque_has_gravity_share_one():
    t = np.arange(int(4 * FS)) / FS
    q = 0.8 + 0.6 * np.sin(2 * np.pi * 0.25 * t)
    tau = 4.0 * G0 * 0.35 * np.sin(q)
    G, why = passivity.gravity_share(tau, tau.copy())
    assert G == pytest.approx(1.0) and why == ""


def test_viscous_torque_has_gravity_share_zero():
    t = np.arange(int(4 * FS)) / FS
    tau = 20.0 * np.cos(2 * np.pi * 0.5 * t)
    G, _ = passivity.gravity_share(tau, np.zeros_like(tau))
    assert G == pytest.approx(0.0, abs=1e-12)


def test_missing_gravity_torque_reports_nan_with_reason():
    G, why = passivity.gravity_share(np.ones(10), None)
    assert np.isnan(G) and why == "no gravity torque provided"


def test_reversals_respect_deadband_and_minimum_separation():
    t = np.arange(int(4 * FS)) / FS
    w = np.deg2rad(60) * np.sin(2 * np.pi * 1.0 * t)        # 7 interior reversals in 4 s
    w_noisy = w + np.deg2rad(3) * np.sin(2 * np.pi * 40 * t)  # chatter inside the 5 deg/s band
    a = passivity.reversal_metrics(w, FS)
    b = passivity.reversal_metrics(w_noisy, FS)
    assert a["n_reversals"] == 7 and b["n_reversals"] == 7
    assert a["latency_med_s"] == pytest.approx(0.5, abs=0.01)  # peak to peak of a 1 Hz sine
    assert a["alpha_req_med_rad_s2"] == pytest.approx(2 * np.deg2rad(60) / 0.5, rel=0.03)


def test_layer_reports_g_and_reversals_per_dof():
    from frame_core import analyze
    from frame_core.synthetic import make_capture, default_config
    res = analyze(make_capture(n_reps=4), default_config())
    e = res["passivity"]["per_dof"]["right_shoulder_Z"]
    assert set(passivity.METRIC_KEYS) <= set(e)
    assert np.isnan(e["G"]["median"]) and e["G_reason"] == "no gravity torque provided"
    assert e["n_reversals"]["median"] >= 1


def test_spec_scalars_never_touch_time_normalisation(monkeypatch):
    """Identical whatever the curves' resolution is -- and computed without ever resampling."""
    from frame_core import analyze, curves, signal as sig
    from frame_core.synthetic import make_capture, default_config
    cap = make_capture(n_reps=6)
    ref = analyze(cap, default_config())

    def boom(*a, **k):
        raise AssertionError("a spec scalar was computed from a time-normalised signal")

    monkeypatch.setattr(sig, "resample_interp", boom)
    monkeypatch.setattr(curves, "resample_interp", boom)
    monkeypatch.setattr(curves, "trial_curves", boom)
    again = analyze(cap, default_config())
    for d, e in ref["passivity"]["per_dof"].items():
        for k in passivity.METRIC_KEYS:
            a, b = e[k]["median"], again["passivity"]["per_dof"][d][k]["median"]
            assert (np.isnan(a) and np.isnan(b)) or a == b
