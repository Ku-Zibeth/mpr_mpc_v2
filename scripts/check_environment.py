#!/usr/bin/env python
"""Check the runtime environment required by MPR-MPC."""

from __future__ import annotations

import importlib
import platform
import sys
from argparse import ArgumentParser
from importlib import metadata
from pathlib import Path


SCRIPT = Path(__file__).resolve()
MPR_ROOT = SCRIPT.parents[1]

if str(MPR_ROOT) not in sys.path:
    sys.path.insert(0, str(MPR_ROOT))

from _bootstrap import (  # noqa: E402
    LOG_ROOT,
    PACKAGE_ROOT,
    TDMPC2_RUNTIME,
    VENDOR_ROOT,
    bootstrap,
)


bootstrap()


EXPECTED = {
    "torch": "2.7.1",
    "metadrive-simulator": "0.4.3",
    "gymnasium": "0.29.1",
    "hydra-core": "1.3.2",
    "omegaconf": "2.3.0",
    "tensordict": "0.8.3",
    "torchrl": "0.8.1",
}


def ok(name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[OK] {name}{suffix}")


def fail(name: str, detail: str) -> None:
    print(f"[FAIL] {name} - {detail}")
    raise SystemExit(1)


def warn(name: str, detail: str) -> None:
    print(f"[WARN] {name} - {detail}")


def dist_version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError as exc:
        fail(dist, "package is not installed")
        raise AssertionError from exc


def import_module(module: str, dist: str | None = None) -> object:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report exact import failure to user
        fail(module, repr(exc))
    version = dist_version(dist) if dist else getattr(mod, "__version__", "imported")
    ok(module, str(version))
    return mod


def check_dist(dist: str) -> str:
    version = dist_version(dist)
    expected = EXPECTED.get(dist)
    if expected is not None and version != expected:
        warn(dist, f"verified version is {expected}, installed version is {version}")
    else:
        ok(dist, version)
    return version


def module_file(module: object) -> Path:
    filename = getattr(module, "__file__", None)
    if not filename:
        fail(getattr(module, "__name__", repr(module)), "module has no __file__")
    return Path(str(filename)).resolve()


def assert_under(label: str, module: object, root: Path) -> None:
    path = module_file(module)
    try:
        path.relative_to(root.resolve())
    except ValueError:
        fail(label, f"external runtime dependency detected: {path}")
    ok(f"{label} source", str(path))


def full_metadrive_check() -> None:
    from omegaconf import OmegaConf

    from common.parser import parse_cfg
    from env import make_env

    cfg = OmegaConf.load(MPR_ROOT / "config.yaml")
    cfg.enable_wandb = False
    cfg.save_agent = False
    cfg.save_csv = False
    cfg.save_video = False
    cfg.compile = False
    cfg.log_root = str(LOG_ROOT)
    cfg.metadrive.simulator.use_render = False
    cfg.metadrive.simulator.horizon = 10
    cfg = parse_cfg(cfg)
    env = make_env(cfg)
    try:
        env.reset(seed=int(cfg.metadrive["simulator"]["start_seed"]))
        checked_steps = 0
        for _ in range(3):
            _, reward, done, info = env.step(env.rand_act())
            values = {"reward": float(reward)}
            for key in ("cost", "risk_field_cost"):
                if key not in info:
                    fail("MetaDrive full check", f"info missing {key!r}")
                values[key] = float(info[key])
            checked_steps += 1
            if done:
                break
    finally:
        env.close()
    ok("MetaDrive full reset/step", f"{checked_steps} step(s)")


def main(*, full: bool = False) -> None:
    version = sys.version_info
    if version.major != 3 or version.minor != 11:
        fail("Python", f"expected 3.11.x, got {platform.python_version()}")
    ok("Python", platform.python_version())
    ok("Repository root", str(PACKAGE_ROOT.resolve()))
    ok("Default log root", str(LOG_ROOT.resolve()))

    check_dist("torch")
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - report exact import failure to user
        fail("torch", repr(exc))
    cuda_runtime = str(torch.version.cuda)
    if cuda_runtime != "12.6":
        warn("PyTorch CUDA runtime", f"verified runtime is 12.6, installed is {torch.version.cuda}")
    else:
        ok("PyTorch CUDA runtime", cuda_runtime)
    if not torch.cuda.is_available():
        fail("CUDA", "torch.cuda.is_available() is False")
    ok("CUDA", torch.cuda.get_device_name(0))
    ok("CUDA device count", str(torch.cuda.device_count()))

    for dist in (
        "metadrive-simulator",
        "gymnasium",
        "hydra-core",
        "omegaconf",
        "tensordict",
        "torchrl",
    ):
        check_dist(dist)
    import_module("numpy", "numpy")
    import_module("hydra")
    import_module("omegaconf")
    import_module("wandb", "wandb")
    import_module("gymnasium")
    import_module("metadrive")
    import_module("tensordict")
    import_module("torchrl")
    import_module("pandas", "pandas")
    import_module("scipy", "scipy")

    env_mod = import_module("env")
    common_buffer = import_module("common.buffer")
    lattice_mod = import_module("lattice")
    lattice_impl = import_module("lattice.frenet_metadrive")
    lattice_tdmpc2_mod = import_module("lattice_tdmpc2")
    tdmpc2_mod = import_module("tdmpc2")
    import_module("mpr_mpc")
    import_module("mpr_mpc.agent")
    import_module("mpr_mpc.planning.coordinator")
    import_module("mpr_mpc.residual_rl")
    core_mod = import_module("mpr_mpc.tdmpc2.core")

    assert_under("env", env_mod, PACKAGE_ROOT)
    assert_under("common.buffer", common_buffer, TDMPC2_RUNTIME)
    assert_under("lattice", lattice_mod, PACKAGE_ROOT)
    assert_under("lattice.frenet_metadrive", lattice_impl, PACKAGE_ROOT)
    assert_under("lattice_tdmpc2", lattice_tdmpc2_mod, PACKAGE_ROOT)
    assert_under("tdmpc2", tdmpc2_mod, TDMPC2_RUNTIME)
    assert_under("mpr_mpc.tdmpc2.core", core_mod, PACKAGE_ROOT)
    ok("vendor root", str(VENDOR_ROOT.resolve()))
    ok("TDMPC2 class module", core_mod.MPRTDMPC2.__mro__[1].__module__)

    config_path = MPR_ROOT / "config.yaml"
    if not config_path.is_file():
        fail("MPR-MPC config", f"missing {config_path}")
    text = config_path.read_text(encoding="utf-8")
    expected = "algorithm_version: mpr_mpc_residual_sac_mppi_v2"
    if expected not in text:
        fail("algorithm_version", f"expected line not found in {config_path}")
    ok("algorithm_version", "mpr_mpc_residual_sac_mppi_v2")

    if full:
        full_metadrive_check()

    print("Environment check passed.")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Create a MetaDrive environment and run a short reset/step sanity check.",
    )
    args = parser.parse_args()
    main(full=bool(args.full))
