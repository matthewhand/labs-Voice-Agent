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

"""Config-driven enable/disable of the LLM and TTS services.

Building an enabled service eagerly downloads/loads models and needs a GPU, so
these tests only ever exercise the disabled path on GPU-less CI.
"""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_voice_agent.pipecat.services.nemo.builders import (
    build_llm,
    build_llm_text_processor,
    build_tts,
)
from nemo_voice_agent.utils.config_manager import ConfigManager


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_BASE_PATH = os.path.join(REPO_ROOT, "examples", "generic_voice_agent", "server")
SERVER_CONFIGS_PATH = os.path.join(SERVER_BASE_PATH, "server_configs")
AMD_ROCM_CONFIG = os.path.join(SERVER_CONFIGS_PATH, "amd_rocm.yaml")
DEFAULT_CONFIG = os.path.join(SERVER_CONFIGS_PATH, "default.yaml")


@pytest.fixture
def amd_rocm_config_manager():
    return ConfigManager(server_base_path=SERVER_BASE_PATH, server_config_path=AMD_ROCM_CONFIG)


class TestDisabledServicesFlags:
    """USE_LLM / USE_TTS / USE_DIAR flags derived from the server config."""

    @pytest.mark.unit
    def test_amd_rocm_disables_llm_and_tts_but_keeps_diar(self, amd_rocm_config_manager):
        config_manager = amd_rocm_config_manager
        assert config_manager.USE_LLM is False
        assert config_manager.USE_TTS is False
        assert config_manager.USE_DIAR is True
        assert config_manager.SYSTEM_ROLE == "system"
        assert config_manager.SYSTEM_PROMPT

    @pytest.mark.unit
    def test_default_enables_llm_and_tts(self):
        config_manager = ConfigManager(server_base_path=SERVER_BASE_PATH, server_config_path=DEFAULT_CONFIG)
        assert config_manager.USE_LLM is True
        assert config_manager.USE_TTS is True

    @pytest.mark.unit
    def test_missing_llm_and_tts_sections_disable_services(self, tmp_path):
        """Sections entirely absent must behave like ``enabled: false``."""
        config = OmegaConf.load(AMD_ROCM_CONFIG)
        del config["llm"]
        del config["tts"]
        config_path = tmp_path / "no_llm_no_tts.yaml"
        OmegaConf.save(config, config_path)

        config_manager = ConfigManager(server_base_path=SERVER_BASE_PATH, server_config_path=str(config_path))
        assert config_manager.USE_LLM is False
        assert config_manager.USE_TTS is False


class TestDisabledServiceBuilders:
    """Builders short-circuit to ``None`` without loading any model."""

    @pytest.mark.unit
    def test_amd_rocm_builders_return_none(self, amd_rocm_config_manager):
        config_manager = amd_rocm_config_manager
        assert build_llm(config_manager) is None
        assert build_tts(config_manager) is None
        assert build_llm_text_processor(config_manager) is None

    @pytest.mark.unit
    def test_llm_text_processor_guards_on_use_tts(self, amd_rocm_config_manager):
        assert build_llm_text_processor(amd_rocm_config_manager) is None
