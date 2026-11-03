"""Content-addressed prediction origins for real counterfactual replays."""

import hashlib
import json

import numpy as np


def make_anchor(scene_id, seed, prefix_actions, observation, fingerprint, control_dt):
    payload = dict(scene_id=int(scene_id), seed=int(seed),
                   prefix_actions=np.asarray(prefix_actions, dtype=float).reshape(-1, 2).tolist(),
                   observation=np.asarray(observation, dtype=float).tolist(),
                   fingerprint=np.asarray(fingerprint, dtype=float).tolist(),
                   prediction_step=len(prefix_actions), control_dt=float(control_dt))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def verify_anchor(anchor_id, payload):
    expected, canonical = make_anchor(payload["scene_id"], payload["seed"], payload["prefix_actions"],
        payload["observation"], payload["fingerprint"], payload["control_dt"])
    if expected != anchor_id or payload != canonical:
        raise ValueError("Calibration anchor content/hash mismatch.")
