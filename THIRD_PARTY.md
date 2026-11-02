# Third-Party And Vendored Runtime

This repository vendors the runtime source required by the tested MPR-MPC
MetaDrive training path so a clone of `Ku-Zibeth/MPR_MetaDrive` can run without
an additional local TD-MPC2 checkout.

## TD-MPC2 Runtime

Source:

```text
https://github.com/nicklashansen/tdmpc2
```

Base commit of the outer working tree used for vendoring:

```text
e9f59321933cbc8e11a002b842adc7d4ffae8ff1
```

Vendored location:

```text
vendor/tdmpc2_runtime/
```

License:

```text
vendor/tdmpc2_runtime/LICENSE
```

The vendored runtime includes local project-specific changes from the inspected
working tree. The modified TD-MPC2 files observed before vendoring were:

```text
tdmpc2/common/buffer.py
tdmpc2/common/layers.py
tdmpc2/common/logger.py
tdmpc2/common/world_model.py
tdmpc2/config.yaml
tdmpc2/tdmpc2.py
tdmpc2/trainer/online_trainer.py
```

These local changes are required by the current MPR-MPC MetaDrive training path
for cost fields, logging/checkpoint behavior, CUDA replay storage, and runtime
compatibility with the staged MPR-MPC trainer.

## Lattice Runtime

Source:

```text
local project-specific lattice implementation from the tested TD-MPC2 workspace
```

Vendored location:

```text
lattice/
```

This code provides the Frenet/Lattice planner imported as `lattice.*` by
MPR-MPC and by `lattice_tdmpc2`.

Exact upstream commit metadata for this project-specific Lattice snapshot is not
available in the inspected workspace. This vendored snapshot is therefore the
canonical Lattice runtime used by MPR-MPC. It was copied from the tested local
TD-MPC2 workspace on 2026-09-27 to make `MPR_MetaDrive` self-contained.

## lattice_tdmpc2

Source:

```text
https://github.com/Ku-Zibeth/lattice_tdmpc2
```

Commit:

```text
638fa40cffadf547d025f5aecbad5ffe97a6aad7
```

Vendored location:

```text
lattice_tdmpc2/
```

The inspected local `lattice_tdmpc2` checkout did not contain a separate
`LICENSE` file. Keep upstream attribution and commit metadata when redistributing
or publishing this vendored copy.

## MetaDrive

MetaDrive is not vendored. It is installed from PyPI:

```text
metadrive-simulator==0.4.3
```
