"""Restore parameter checkpoints on the current machine's default device."""

from pathlib import Path

import jax
import orbax.checkpoint as ocp


def restore_params(path: str | Path):
    """Avoid restoring the original machine's GPU/TPU sharding metadata."""
    path = str(Path(path).expanduser().resolve())
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    with ocp.StandardCheckpointer() as checkpointer:
        target = jax.tree_util.tree_map(
            lambda leaf: jax.ShapeDtypeStruct(
                leaf.shape,
                leaf.dtype,
                sharding=sharding,
            ),
            checkpointer.metadata(path),
        )
        return checkpointer.restore(path, args=ocp.args.StandardRestore(target))
