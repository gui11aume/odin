"""Two-part architecture checksum (configid) for the Odin model.

The full reference of a released model is `<family>.<wiring>.<layout>.<gen>`,
e.g. `odin.7K2.M9Q.1`. The final `<gen>` (weight generation) is assigned
manually when a model is promoted for real use; it is not part of the
checksum. During development, models keep their version names (v22, v23, ...).

The checksum has two independently hashed parts:

* **wiring** — which model it is: the architectural fields of
  :class:`~odin_vae.config_classes.ConfigForModel` (`ARCH_FIELDS`) plus the
  architectural constants hardcoded in :class:`~odin_vae.model.OdinModel`
  (`ARCH_CONSTANTS`). A change here means the computation graph changed.
* **layout** — what the weights look like: the `state_dict` shape map
  (parameter/buffer name to shape, sorted). It captures everything that
  defines the weight layout, including `vocab_size`, which is not a config
  field. Parameter names come from the modeling library, so a library upgrade
  that renames parameters changes the layout part.

Reading the two parts against each other:

* same wiring, different layout: re-instantiation (e.g. new tokenizer/vocab);
* different wiring, same layout: wiring tweak that does not change tensor
  shapes (e.g. local-attention window, weight tying, T5 rel buckets);
* both different: new architecture.

The training harness stamps `configid` into each checkpoint's
`hyper_parameters`; `verify_configid` uses the stamp to check the wiring
part as well as the layout (older checkpoints without the stamp get the
layout check only, since a state dict carries no wiring).

The scheme is family-agnostic and meant to be reused verbatim for other
models: the family name is not hashed. Bump `SPEC_VERSION` to invalidate
all previously issued ids (the safe direction).

Scheme definition (reproducible outside this repo):

* wiring spec: the line `SPEC_VERSION`, then one line `F <field>=<value>`
  per entry of `ARCH_FIELDS` (sorted by field name, Python `str()` of the
  value), then one line `C <name>=<value>` per entry of `ARCH_CONSTANTS`
  (sorted by name).
* layout spec: the line `SPEC_VERSION`, then one line
  `S <name>=<d0,d1,...>` per state-dict key (sorted by key name).
* each part id: take the BLAKE2b digest of the spec (`digest_size=2`,
  big-endian integer), drop the least significant bit (15 bits), encode as
  3 Crockford base32 characters (`0-9A-V`, no I/L/O/U), most significant
  digit first.
"""

from __future__ import annotations

import hashlib
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from .config_classes import ConfigForModel


class _StateDictProvider(Protocol):
    """Anything with a `state_dict()` (e.g. `torch.nn.Module`)."""

    def state_dict(self) -> Mapping[str, torch.Tensor]: ...


class _ConfiguredStateDictProvider(_StateDictProvider, Protocol):
    """A model that also carries its :class:`ConfigForModel` (e.g. `OdinModel`)."""

    config: ConfigForModel


#: Bump to invalidate every previously issued id (scheme-level, not per-model).
SPEC_VERSION = "archid/1"

#: Architectural fields of ConfigForModel, in no particular order (the spec
#: sorts them). Deliberately excluded: tokenizer_path (a path is not an
#: identity; the vocab size is captured by the layout part), use_fast_tokenizer
#: (implementation detail), kl_weight and num_latent_samples (training, not
#: architecture).
ARCH_FIELDS: tuple[str, ...] = (
    "attention_heads",
    "decoder",
    "decoder_layers",
    "encoder_layers",
    "hidden_size",
    "intermediate_size",
    "local_attention",
    "max_position_embeddings",
)

#: Architectural constants hardcoded in model.py (not in ConfigForModel).
#: Keep in sync with OdinModel.__init__.
ARCH_CONSTANTS: dict[str, Any] = {
    "encoder_pooling": "pma_last_nonpad",
    "t5_rel_buckets": 32,
    "tie_embeddings": True,
}

#: Crockford base32 alphabet: 0-9 and A-Z minus the visually ambiguous I, L, O, U.
_CROCKFORD = "".join(c for c in "0123456789" + string.ascii_uppercase if c not in "ILOU")


def _crockford15(spec: str) -> str:
    """15 bits of BLAKE2b(spec) as 3 Crockford base32 characters."""
    n = int.from_bytes(hashlib.blake2b(spec.encode("utf-8"), digest_size=2).digest(), "big") >> 1
    return "".join(_CROCKFORD[(n >> (10 - 5 * i)) & 31] for i in range(3))


def wiring_spec(cfg: ConfigForModel) -> str:
    """Canonical text of the wiring block (see module docstring)."""
    lines = [SPEC_VERSION]
    lines += [f"F {k}={getattr(cfg, k)}" for k in sorted(ARCH_FIELDS)]
    lines += [f"C {k}={v}" for k, v in sorted(ARCH_CONSTANTS.items())]
    return "\n".join(lines)


def layout_spec(shape_map: Mapping[str, Sequence[int]]) -> str:
    """Canonical text of the layout block (see module docstring)."""
    lines = [SPEC_VERSION]
    lines += [f"S {k}={','.join(str(d) for d in shape)}" for k, shape in sorted(shape_map.items())]
    return "\n".join(lines)


def wiring_id(cfg: ConfigForModel) -> str:
    """3-char id of the wiring block."""
    return _crockford15(wiring_spec(cfg))


def layout_id(shape_map: Mapping[str, Sequence[int]]) -> str:
    """3-char id of the layout block."""
    return _crockford15(layout_spec(shape_map))


def shape_map_of(model: _StateDictProvider) -> dict[str, tuple[int, ...]]:
    """Name to shape of every persistent tensor in the model."""
    return {k: tuple(t.shape) for k, t in model.state_dict().items()}


def configid(cfg: ConfigForModel, model: _StateDictProvider) -> str:
    """The full `<wiring>.<layout>` checksum of a live model."""
    return f"{wiring_id(cfg)}.{layout_id(shape_map_of(model))}"


def configid_from_state_dict(cfg: ConfigForModel, state_dict: Mapping[str, torch.Tensor]) -> str:
    """The `<wiring>.<layout>` checksum from a state dict (no model needed)."""
    return f"{wiring_id(cfg)}.{layout_id({k: tuple(t.shape) for k, t in state_dict.items()})}"


def _state_dict_of(ckpt: Any) -> dict:
    """The model state dict of a checkpoint object, `model.` prefix stripped."""
    state = ckpt.get("state_dict", ckpt)
    prefix = "model."
    if state and all(str(k).startswith(prefix) for k in state):
        state = {str(k)[len(prefix) :]: v for k, v in state.items()}
    return state


def _stored_configid(ckpt: Any) -> str | None:
    """The configid stamped in `hyper_parameters` (None for older checkpoints)."""
    hp = ckpt.get("hyper_parameters")
    if isinstance(hp, dict):
        value = hp.get("configid")
        if isinstance(value, str):
            return value
    return None


def configid_of_checkpoint(cfg: ConfigForModel, checkpoint: str | Path) -> str:
    """The `<wiring>.<layout>` checksum of a Lightning checkpoint file.

    The wiring part comes from `cfg` (a state dict carries no wiring); the
    layout part from the state-dict shapes. Strips the `model.` prefix used
    by the training harness checkpoints.
    """
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    return configid_from_state_dict(cfg, _state_dict_of(ckpt))


def verify_configid(model: _ConfiguredStateDictProvider, checkpoint: str | Path) -> str:
    """Check that `checkpoint` matches the architecture of `model`.

    Returns the configid on success; raises `ValueError` naming the
    mismatched part (wiring vs layout) otherwise. Call before loading
    weights so an architecture mismatch fails with a readable message
    instead of a shape error deep in `load_state_dict`.

    The layout part is always checked (state-dict shapes vs the model). The
    wiring part is checked when the checkpoint stores its own configid in
    `hyper_parameters` (harnesses stamped on or after this change); older
    checkpoints get the layout check only, since a state dict carries no
    wiring.
    """
    expected = configid(model.config, model)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)  # nosec B614  # local Lightning checkpoint
    state = _state_dict_of(ckpt)
    actual = configid_from_state_dict(model.config, state)
    exp_w, exp_l = expected.split(".")
    _, act_l = actual.split(".")
    stored = _stored_configid(ckpt)
    if stored is not None and stored.split(".")[0] != exp_w:
        raise ValueError(
            f"configid wiring mismatch: model {exp_w} vs checkpoint {stored.split('.')[0]} "
            f"({checkpoint}); the computation graph differs."
        )
    if act_l != exp_l:
        raise ValueError(
            f"configid layout mismatch: model {exp_l} vs checkpoint {act_l} "
            f"({checkpoint}); the weight layout differs (e.g. vocab size)."
        )
    return expected


__all__ = [
    "ARCH_CONSTANTS",
    "ARCH_FIELDS",
    "SPEC_VERSION",
    "configid",
    "configid_from_state_dict",
    "configid_of_checkpoint",
    "layout_id",
    "layout_spec",
    "shape_map_of",
    "verify_configid",
    "wiring_id",
    "wiring_spec",
]
