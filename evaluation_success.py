"""The retained release's evaluation endpoint; never changes executed actions."""
from __future__ import annotations
import math


def apply_collision_failure(row):
    """Idempotently count native arrival AND whole-episode no collision as success.

    Native termination and complete-episode reward/cost remain unchanged.
    Existing collision failures are counted once through the union definition.
    """
    native=float(row.get('native_success',row['final_success']))
    collision=float(row['episode_crash'])
    if not math.isfinite(native) or not math.isfinite(collision):
        raise ValueError('Success and whole-episode collision evidence must be finite')
    arrived=native>.5;crashed=collision>.5;success=arrived and not crashed
    row.update(native_success=float(arrived),strict_success=float(success),final_success=float(success),
        strict_failure=float(not success),strict_failure_reason='collision' if crashed else ('task_not_completed' if not arrived else ''),
        collision_and_native_success=float(arrived and crashed),collision_and_success=0.)
    return row
