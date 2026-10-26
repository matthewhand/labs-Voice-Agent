# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-neutral device helpers.

ROCm builds of PyTorch deliberately expose AMD GPUs through the ``torch.cuda``
namespace, so the device strings in the server configs ("cuda", "cuda:0") are
correct on both CUDA and ROCm. These helpers add the missing pieces for running
on AMD hardware:

* ``resolve_device`` validates a configured device string and optionally pins it
  to a named physical accelerator (``AMD_DEVICE_TARGET=dgpu|apu``), because the
  ROCm enumeration order is not guaranteed to match the config author's intent.
* ``empty_device_cache`` guards the CUDA-only cache call so a CPU run does not
  depend on a GPU runtime being importable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import torch


#: Environment variable selecting a named physical GPU. Unset means "use the
#: device string from the config verbatim".
DEVICE_TARGET_ENV = "AMD_DEVICE_TARGET"

#: Token-aware regex patterns matched (case-insensitively) against the
#: ROCm-reported device name for each target. Word boundaries keep matches
#: honest: there is no bare "XT" hint, and "raphics" can no longer select a
#: device by accident. ``dgpu`` is the explicit discrete product lines
#: (RX / PRO W / Radeon Pro); ``apu`` is integrated silicon (Ryzen parts, or the
#: generic "Radeon ... Graphics" iGPU name).
_TARGET_NAME_HINTS: dict[str, tuple[re.Pattern[str], ...]] = {
    # "Radeon RX 7900 XTX", "Radeon RX 6800"
    "dgpu": (
        re.compile(r"\brx\b", re.IGNORECASE),
        # "Radeon PRO W7900"
        re.compile(r"\bpro\s+w", re.IGNORECASE),
        # "Radeon Pro Duo", "Radeon Pro V620"
        re.compile(r"\bradeon\s+pro\b", re.IGNORECASE),
    ),
    # "AMD Ryzen AI 9 HX 370"
    "apu": (
        re.compile(r"\bryzen\b", re.IGNORECASE),
        # "AMD Radeon Graphics", "AMD Radeon(TM) Graphics", "Radeon 780M Graphics"
        re.compile(r"\bradeon\b.*\bgraphics\b", re.IGNORECASE),
    ),
}


@dataclass(frozen=True)
class DeviceSelection:
    """Resolved device plus the diagnostic record of what it landed on."""

    device: torch.device
    requested: str
    target: Optional[str]
    index: Optional[int]
    name: str
    gcn_arch: Optional[str]
    note: str = ""


def _visible_gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _describe(index: int) -> tuple[str, Optional[str]]:
    properties = torch.cuda.get_device_properties(index)
    return properties.name, getattr(properties, "gcnArchName", None)


def _matching_targets(name: str) -> set[str]:
    """Return every target class whose patterns match ``name``.

    Patterns are anchored on word boundaries so a device is classified on
    tokens, not on accidental substrings. A name can match more than one class
    (e.g. ``Radeon RX Vega Graphics``); callers must treat that as ambiguous
    rather than guessing which the config author meant.
    """
    return {
        target for target, patterns in _TARGET_NAME_HINTS.items() if any(pattern.search(name) for pattern in patterns)
    }


def _find_target_index(target: str) -> int:
    """Return the visible GPU index whose name matches ``target``."""
    if target not in _TARGET_NAME_HINTS:
        raise ValueError(
            f"Unknown {DEVICE_TARGET_ENV}={target!r}; expected one of {sorted(_TARGET_NAME_HINTS)} or unset"
        )
    count = _visible_gpu_count()
    names = [_describe(index)[0] for index in range(count)]
    classes = [_matching_targets(name) for name in names]

    for index, matched in enumerate(classes):
        if len(matched) > 1:
            raise RuntimeError(
                f"GPU cuda:{index} ({names[index]}) matches multiple {DEVICE_TARGET_ENV} "
                f"classes {sorted(matched)}; disambiguate with HIP_VISIBLE_DEVICES"
            )

    matches = [index for index, matched in enumerate(classes) if target in matched]
    if not matches:
        available = ", ".join(f"cuda:{index} ({names[index]})" for index in range(count))
        raise RuntimeError(
            f"{DEVICE_TARGET_ENV}={target!r} matched no visible GPU. Visible devices: {available or 'none'}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"{DEVICE_TARGET_ENV}={target!r} matched multiple GPUs {matches}; "
            "narrow the visible set with HIP_VISIBLE_DEVICES"
        )
    return matches[0]


def resolve_device(configured: str, *, target: Optional[str] = None, allow_cpu: bool = True) -> DeviceSelection:
    """Resolve a configured device string into a concrete ``torch.device``.

    ``configured`` accepts ``cpu``, ``cuda`` and ``cuda:N``. When ``target`` (or
    ``AMD_DEVICE_TARGET``) names ``dgpu``/``apu``, the returned index is chosen
    by matching the ROCm-reported device name rather than trusting enumeration
    order. ``cuda`` without an index and without a target keeps index 0.
    """
    configured = (configured or "cpu").strip()
    target = (target or os.environ.get(DEVICE_TARGET_ENV) or "").strip() or None

    if configured == "cpu":
        if not allow_cpu:
            raise RuntimeError("cpu device configured but CPU use is not allowed here")
        return DeviceSelection(torch.device("cpu"), configured, target, None, "cpu", None)

    if not configured.startswith("cuda"):
        raise ValueError(f"Unsupported device {configured!r}; expected 'cpu', 'cuda' or 'cuda:N'")

    if _visible_gpu_count() == 0:
        raise RuntimeError(
            f"Device {configured!r} requested but torch.cuda reports no available devices. "
            "On AMD, install the ROCm PyTorch wheel and check ROCm/HIP visibility."
        )

    explicit_index: Optional[int] = None
    if ":" in configured:
        _, _, suffix = configured.partition(":")
        try:
            explicit_index = int(suffix)
        except ValueError as error:
            raise ValueError(f"Malformed device index in {configured!r}") from error

    if target is not None:
        index = _find_target_index(target)
        if explicit_index is not None and explicit_index != index:
            name, _ = _describe(index)
            raise RuntimeError(
                f"Config requests {configured!r} but {DEVICE_TARGET_ENV}={target!r} resolves to "
                f"cuda:{index} ({name}); set them to agree or unset {DEVICE_TARGET_ENV}"
            )
    else:
        index = explicit_index if explicit_index is not None else 0

    if index >= _visible_gpu_count():
        available = ", ".join(f"cuda:{i} ({_describe(i)[0]})" for i in range(_visible_gpu_count()))
        raise RuntimeError(
            f"Device cuda:{index} is not visible (only {_visible_gpu_count()} device(s): {available}). "
            "Check HIP_VISIBLE_DEVICES."
        )

    name, gcn_arch = _describe(index)
    note = ""
    if torch.version.hip:
        note = f"ROCm/HIP {torch.version.hip}"
    return DeviceSelection(torch.device(f"cuda:{index}"), configured, target, index, name, gcn_arch, note)


def log_selection(selection: DeviceSelection, component: str, logger) -> None:
    """Emit one consistent log line per resolved component device."""
    logger.info(
        f"{component}: device={selection.device} name={selection.name} "
        f"arch={selection.gcn_arch or 'n/a'} target={selection.target or 'none'} {selection.note}".rstrip()
    )


def empty_device_cache() -> None:
    """Release cached GPU memory when a GPU backend is present.

    ``torch.cuda.empty_cache`` is the correct call on ROCm builds too, but it
    raises on a CPU-only install where ``torch.cuda`` has no runtime. Callers
    treat this as best-effort cleanup, so a no-op is the right fallback.
    """
    if not hasattr(torch, "cuda"):
        return
    try:
        if not torch.cuda.is_available():
            return
    except Exception:  # pragma: no cover - defensive: half-initialised CUDA builds
        return
    torch.cuda.empty_cache()
