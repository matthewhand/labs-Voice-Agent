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

"""Regression tests for defects found re-reviewing the live-path fix with fresh eyes.

1. The residual partial buffer (up to one trained chunk) was silently dropped on
   VAD stop; entire sub-chunk utterances got no diarization at all.
2. ``reset()`` dereferenced ``self._model`` unguarded and raised for a disabled
   service, which the RTVI reset-context action can legally call.
"""

import asyncio
from types import SimpleNamespace

from pipecat.frames.frames import VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from nemo_voice_agent.pipecat.services.nemo import diar as diar_module
from nemo_voice_agent.pipecat.services.nemo.diar import NemoDiarService


def _bare():
    service = NemoDiarService.__new__(NemoDiarService)
    service._name = "diar-sweep"
    service._enabled = True
    service._queue = asyncio.Queue()
    service._response_queue = asyncio.Queue()
    service._processing_task = None
    service._response_task = None
    service._vad_user_speaking = True
    service._audio_buffer = [b"residual"]
    service._current_speaker_id = 1
    service._has_generated_metrics = False
    service.frames = []

    async def push(frame, direction=None):
        service.frames.append((frame, direction))

    service.push_frame = push
    return service


def test_reset_with_no_model_is_safe():
    """A disabled service has _model None; the RTVI reset action still calls reset()."""
    service = _bare()
    service._model = None
    service.reset()  # must not raise
    assert service._audio_buffer == [] and service._current_speaker_id is None
    assert not service._vad_user_speaking


def test_reset_calls_model_reset_state_when_present():
    service = _bare()
    calls = []

    class Model:
        def reset_state(self):
            calls.append("reset_state")

    service._model = Model()
    service.reset()
    assert calls == ["reset_state"]


def test_vad_stop_queues_residual_buffer_before_flush(monkeypatch):
    """Trailing partial audio reaches the diarizer before the stop frame is forwarded."""
    service = _bare()
    service._processing_running = True  # consumer live: residual must be flushed
    service._params = SimpleNamespace(buffer_size=30)
    queued = []

    async def capture_put(item):
        queued.append(item)

    service._queue.put = capture_put

    async def no_flush():
        pass

    monkeypatch.setattr(service, "_flush_pending_audio", no_flush)

    async def base(self, frame, direction):
        service.frames.append(("forwarded", type(frame).__name__))

    monkeypatch.setattr(diar_module.STTService, "process_frame", base)

    asyncio.run(service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM))
    assert queued == [b"residual"], "residual buffer must be queued on VAD stop"
    assert service._audio_buffer == [], "residual buffer must be cleared after queueing"
    assert service.frames[-1] == ("forwarded", "VADUserStoppedSpeakingFrame")
