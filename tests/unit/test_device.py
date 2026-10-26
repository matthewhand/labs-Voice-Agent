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

"""Unit tests for the backend-neutral device helpers in ``nemo.device``.

Every test fakes ``torch.cuda`` so no real GPU (CUDA or ROCm) is required.
"""

from unittest.mock import MagicMock

import pytest
import torch

from nemo_voice_agent.pipecat.services.nemo import device


class _FakeDeviceProperties:
    """Minimal stand-in for ``torch.cuda.get_device_properties`` results."""

    def __init__(self, name: str, gcn_arch: str | None = "gfx1100") -> None:
        self.name = name
        self.gcnArchName = gcn_arch


def _fake_cuda(monkeypatch, names):
    """Patch ``torch.cuda`` to report exactly the given device names."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(names))

    def get_device_properties(index):
        return _FakeDeviceProperties(names[index])

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)


def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture(autouse=True)
def _clean_target_env(monkeypatch):
    """Keep the ambient AMD_DEVICE_TARGET from leaking into any test."""
    monkeypatch.delenv(device.DEVICE_TARGET_ENV, raising=False)


APU_NAME = "AMD Radeon 780M Graphics"
DGPU_NAME = "AMD Radeon RX 7900 XTX"


def test_resolve_cpu_returns_cpu_device():
    selection = device.resolve_device("cpu")
    assert selection.device == torch.device("cpu")
    assert selection.index is None
    assert selection.name == "cpu"


def test_resolve_cpu_disallowed_raises():
    with pytest.raises(RuntimeError, match="not allowed"):
        device.resolve_device("cpu", allow_cpu=False)


def test_resolve_unknown_device_string_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported device"):
        device.resolve_device("garbage")


def test_resolve_cuda_without_gpu_raises(monkeypatch):
    _no_cuda(monkeypatch)
    with pytest.raises(RuntimeError, match="no available devices"):
        device.resolve_device("cuda")


def test_resolve_cuda_index_out_of_range_raises(monkeypatch):
    _fake_cuda(monkeypatch, ["AMD Radeon RX 6800"])
    with pytest.raises(RuntimeError, match="not visible"):
        device.resolve_device("cuda:2")


def test_resolve_cuda_index_returns_named_device(monkeypatch):
    _fake_cuda(monkeypatch, ["AMD Radeon RX 6800", DGPU_NAME])
    selection = device.resolve_device("cuda:1")
    assert selection.device == torch.device("cuda:1")
    assert selection.index == 1
    assert selection.name == DGPU_NAME


@pytest.mark.parametrize(
    "names",
    [
        [APU_NAME, DGPU_NAME],
        [DGPU_NAME, APU_NAME],
    ],
)
def test_target_dgpu_picks_rx_regardless_of_order(monkeypatch, names):
    _fake_cuda(monkeypatch, names)
    selection = device.resolve_device("cuda", target="dgpu")
    assert selection.index == names.index(DGPU_NAME)
    assert selection.name == DGPU_NAME
    assert selection.device == torch.device(f"cuda:{selection.index}")


def test_target_apu_picks_integrated_graphics(monkeypatch):
    _fake_cuda(monkeypatch, [DGPU_NAME, APU_NAME])
    selection = device.resolve_device("cuda", target="apu")
    assert selection.index == 1
    assert selection.name == APU_NAME


def test_unknown_amd_device_target_raises_value_error(monkeypatch):
    _fake_cuda(monkeypatch, [DGPU_NAME])
    monkeypatch.setenv(device.DEVICE_TARGET_ENV, "tpu")
    with pytest.raises(ValueError, match="Unknown AMD_DEVICE_TARGET"):
        device.resolve_device("cuda")


def test_target_conflicting_with_explicit_index_raises(monkeypatch):
    _fake_cuda(monkeypatch, [APU_NAME, DGPU_NAME])
    with pytest.raises(RuntimeError, match="resolves to"):
        device.resolve_device("cuda:0", target="dgpu")


def test_device_matching_both_classes_raises_ambiguity(monkeypatch):
    _fake_cuda(monkeypatch, ["AMD Radeon RX Vega Graphics"])
    with pytest.raises(RuntimeError, match="matches multiple"):
        device.resolve_device("cuda", target="dgpu")


def test_empty_device_cache_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    device.empty_device_cache()
    empty_cache.assert_not_called()


def test_empty_device_cache_calls_torch_when_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    device.empty_device_cache()
    empty_cache.assert_called_once_with()
