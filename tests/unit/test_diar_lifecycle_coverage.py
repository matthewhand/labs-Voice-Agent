# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mocked lifecycle coverage for the NeMo diarization service."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import ErrorFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language

from nemo_voice_agent.pipecat.services.nemo import diar as diar_module
from nemo_voice_agent.pipecat.services.nemo.diar import NemoDiarService


def _params():
    return SimpleNamespace(
        frame_len_in_secs=0.16,
        raw_audio_frame_len_in_secs=0.08,
        buffer_size=2,
        threshold=0.5,
        language=Language.EN_US,
    )


def test_constructor_disabled_legacy_and_invalid_backend(monkeypatch):
    monkeypatch.setattr(diar_module.STTService, "__init__", lambda self, **kwargs: None)
    disabled = NemoDiarService(model="", params=_params(), enabled=True, use_vad=False)
    assert disabled._model is None and not disabled._enabled and disabled._vad_user_speaking

    made = {}

    class Model:
        def __init__(self, config, model, **kwargs):
            made.update(config=config, model=model, kwargs=kwargs)

    monkeypatch.setattr(diar_module, "NeMoStreamingDiarService", Model)
    legacy = NemoDiarService(model="diar.nemo", params=_params(), enabled=True, device="cpu")
    assert isinstance(legacy._model, Model) and made["model"] == "diar.nemo"
    assert made["kwargs"]["frame_len_in_secs"] == 0.16

    with pytest.raises(ValueError, match="Invalid backend"):
        NemoDiarService(model="diar.nemo", params=_params(), enabled=True, backend="other")


def _bare():
    service = NemoDiarService.__new__(NemoDiarService)
    service._name = "diar-test"
    service._enabled = True
    service._params = _params()
    service._queue = asyncio.Queue()
    service._response_queue = asyncio.Queue()
    service._processing_task = None
    service._response_task = None
    service._vad_user_speaking = False
    service._audio_buffer = []
    service._current_speaker_id = 2
    service._has_generated_metrics = False
    service.frames = []

    async def push(frame, direction=None):
        service.frames.append((frame, direction))

    async def noop(*args, **kwargs):
        return None

    service.push_frame = push
    service.stop_ttfb_metrics = noop
    service.stop_processing_metrics = noop
    return service


def test_start_stop_cancel_and_language_lifecycle(monkeypatch):
    calls = []

    async def base(self, frame=None):
        calls.append("base")

    monkeypatch.setattr(diar_module.STTService, "start", base)
    monkeypatch.setattr(diar_module.STTService, "stop", base)
    monkeypatch.setattr(diar_module.STTService, "cancel", base)
    service = _bare()
    service._load_model = lambda: calls.append("load")
    service.create_task = lambda coroutine: (coroutine.close(), SimpleNamespace())[1]

    async def cancel_task(task):
        calls.append("cancel-task")

    service.cancel_task = cancel_task
    asyncio.run(service.start(object()))
    assert service._processing_task is not None and service._response_task is not None
    asyncio.run(service.stop(object()))
    assert service._processing_task is None and service._response_task is None
    assert asyncio.run(service._queue.get()) is None
    service._processing_task = service._response_task = SimpleNamespace()
    asyncio.run(service.cancel(object()))
    assert calls.count("cancel-task") == 4

    asyncio.run(service.set_language(Language.FR))
    assert service._params.language == Language.FR
    service._params = None
    asyncio.run(service.set_language(Language.EN))
    assert service._params.language == Language.EN


def test_response_handler_emits_errors_and_success_then_cancels():
    service = _bare()
    handled = []

    async def handle(result):
        handled.append(result)

    service._handle_diarization_result = handle

    class Results:
        def __init__(self):
            self.values = iter([("error", RuntimeError("bad diar")), "speaker"])

        async def get(self):
            try:
                return next(self.values)
            except StopIteration:
                raise asyncio.CancelledError from None

    service._response_queue = Results()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service._response_task_handler())
    assert isinstance(service.frames[0][0], ErrorFrame)
    assert handled == ["speaker"] and service._has_generated_metrics


def test_process_frame_tracks_vad_and_disabled_passthrough(monkeypatch):
    service = _bare()

    async def base(self, frame, direction):
        service.frames.append(("base", direction))

    monkeypatch.setattr(diar_module.STTService, "process_frame", base)
    start = VADUserStartedSpeakingFrame()
    stop = VADUserStoppedSpeakingFrame()
    asyncio.run(service.process_frame(start, FrameDirection.DOWNSTREAM))
    assert service._vad_user_speaking and service._audio_buffer == []
    service._audio_buffer = [b"audio"]
    asyncio.run(service.process_frame(stop, FrameDirection.UPSTREAM))
    assert not service._vad_user_speaking and service._current_speaker_id is None

    service._enabled = False
    asyncio.run(service.process_frame(start, FrameDirection.DOWNSTREAM))
    assert service.frames[-1] == (start, FrameDirection.DOWNSTREAM)


def test_buffer_size_defaults_to_trained_chunk_and_respects_override(monkeypatch):
    monkeypatch.setattr(diar_module.STTService, "__init__", lambda self, **kwargs: None)

    class Model:
        def __init__(self, config, model, **kwargs):
            pass

    monkeypatch.setattr(diar_module, "NeMoStreamingDiarService", Model)

    defaults = _params()
    defaults.frame_len_in_secs = 0.08
    defaults.raw_audio_frame_len_in_secs = 0.016
    defaults.buffer_size = None
    NemoDiarService(model="diar.nemo", params=defaults, enabled=True, device="cpu")
    # Trained chunk: 6 * 80ms = 0.48s = 30 raw 16ms frames.
    assert defaults.buffer_size == 30

    overridden = _params()
    overridden.frame_len_in_secs = 0.08
    overridden.raw_audio_frame_len_in_secs = 0.016
    overridden.buffer_size = 12
    NemoDiarService(model="diar.nemo", params=overridden, enabled=True, device="cpu")
    assert overridden.buffer_size == 12


def test_flush_holds_vad_stop_until_queue_drained(monkeypatch):
    """The VAD stop frame is forwarded only after queued audio is diarized and pushed."""
    service = _bare()
    service._params = _params()  # buffer_size=1: every run_stt chunk goes straight to the queue

    events = []

    async def base(self, frame, direction):
        events.append(("forwarded", type(frame).__name__))

    monkeypatch.setattr(diar_module.STTService, "process_frame", base)

    release = threading.Event()

    class Model:
        def diarize(self, audio):
            release.wait(5)
            return "pred"

    service._model = Model()

    async def scenario():
        loop = asyncio.get_running_loop()
        service.get_event_loop = lambda: loop
        service.create_task = lambda coro: loop.create_task(coro)
        service._processing_running = True
        processor = loop.create_task(service._processing_task_handler())
        responder = loop.create_task(service._response_task_handler())

        async def handle(result):
            events.append(("result", result))

        service._handle_diarization_result = handle

        await service._queue.put(b"audio")
        stop = VADUserStoppedSpeakingFrame()
        held = loop.create_task(service.process_frame(stop, FrameDirection.UPSTREAM))
        await asyncio.sleep(0.1)
        # Diarization is still blocked inside the model; the stop frame must be held.
        assert events == []

        release.set()
        await asyncio.wait_for(held, 5)
        # The diarization result was pushed, and only then the stop frame forwarded.
        assert events == [("result", "pred"), ("forwarded", "VADUserStoppedSpeakingFrame")]

        await service._queue.put(None)
        await asyncio.sleep(0.1)
        processor.cancel()
        responder.cancel()
        await asyncio.gather(processor, responder, return_exceptions=True)

    asyncio.run(scenario())
