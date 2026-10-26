# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Model-free unit tests for the diarization verification scripts.

The ``scripts/*.py`` helpers are plain modules, so they are loaded by path with
``importlib`` (``scripts/`` is not a package). The heavy
``nemo_voice_agent...streaming_diar`` import - which drags in torch and NeMo -
is stubbed while they are executed, so this file needs no model, no GPU and no
network, and only the *pure* scoring logic is exercised.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import math
import re
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import websockets
from pipecat.serializers.protobuf import MessageFrame, ProtobufFrameSerializer


pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_DIAR_PY = Path(__file__).resolve().parents[2] / "nemo_voice_agent/pipecat/services/nemo/diar.py"
_STREAMLING_DIAR = "nemo_voice_agent.pipecat.services.nemo.streaming_diar"
STEP = 0.08


# --------------------------------------------------------------------------- #
# loading the scripts without torch/NeMo
# --------------------------------------------------------------------------- #
class _StubConfig:
    def __init__(self) -> None:
        self.device = "cpu"


class _StubService:
    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("the verification scripts must not build a model in a unit test")


@contextlib.contextmanager
def _stubbed_streaming_diar():
    stub = types.ModuleType(_STREAMLING_DIAR)
    stub.DiarizationConfig = _StubConfig
    stub.NeMoStreamingDiarService = _StubService
    saved = sys.modules.get(_STREAMLING_DIAR)
    sys.modules[_STREAMLING_DIAR] = stub
    try:
        yield stub
    finally:
        if saved is None:
            sys.modules.pop(_STREAMLING_DIAR, None)
        else:
            sys.modules[_STREAMLING_DIAR] = saved


def _load_script(filename: str, alias: str) -> types.ModuleType:
    cached = sys.modules.get(alias)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(alias, _SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    with _stubbed_streaming_diar():
        spec.loader.exec_module(module)
    return module


verify = _load_script("verify_diarization.py", "diar_verify_diarization")
sweep = _load_script("sweep_diarization.py", "diar_sweep_diarization")
endpoint = _load_script("verify_diarization_endpoint.py", "diar_verify_diarization_endpoint")


# --------------------------------------------------------------------------- #
# _majority
# --------------------------------------------------------------------------- #
def test_majority_handles_empty_and_all_silence():
    assert sweep._majority([]) is None
    assert sweep._majority([None, None, None]) is None


def test_majority_single_label_and_ignores_silence():
    assert sweep._majority([3, 3, None, 3]) == 3
    assert sweep._majority([None, 5, None, 5, None]) == 5


def test_majority_tie_breaks_to_lowest_index_like_argmax():
    # np.argmax(bincount([1, 0, 1, 0])) == 0
    assert sweep._majority([0, 1, 0, 1]) == 0
    assert sweep._majority([2, 0, 2]) == 2


# --------------------------------------------------------------------------- #
# _score
# --------------------------------------------------------------------------- #
def _timeline(segments: list[tuple[int, int, int | None]], length: int = 0) -> list[int | None]:
    """Build a frame-indexed timeline; ``segments`` is [(first, stop_exclusive, label)]."""
    length = max([length, *(stop for _, stop, _ in segments)])
    timeline: list[int | None] = [None] * length
    for first, stop, label in segments:
        for index in range(first, stop):
            timeline[index] = label
    return timeline


# silence(1s) + turn0(1s) + silence + turn1(1s) + silence + turn2(1s) + silence == 7.0s == 88 frames.
THREE_SPEAKER_WINDOWS = [(1.0, 2.0, 0), (3.0, 4.0, 1), (5.0, 6.0, 2)]
THREE_SPEAKER_TIMELINE = _timeline(
    [
        (13, 25, 0),  # 1.04s .. 2.00s
        (38, 50, 1),  # 3.04s .. 4.00s
        (63, 75, 2),  # 5.04s .. 6.00s
    ],
    length=88,
)


def test_score_passes_a_correct_three_speaker_timeline():
    assert len(THREE_SPEAKER_TIMELINE) == 88
    result = sweep._score(THREE_SPEAKER_TIMELINE, THREE_SPEAKER_WINDOWS, speakers=3)
    assert result["pass"] is True
    assert result["per_turn"] == [0, 1, 2]
    assert result["distinct"] == 3


def test_score_accepts_a_label_permutation():
    permuted = _timeline([(13, 25, 2), (38, 50, 0), (63, 75, 1)], length=88)
    result = sweep._score(permuted, THREE_SPEAKER_WINDOWS, speakers=3)
    assert result["pass"] is True
    assert result["per_turn"] == [2, 0, 1]


def test_score_fails_when_two_turns_merge_into_one_speaker():
    merged = _timeline([(13, 25, 0), (38, 50, 0), (63, 75, 2)])
    result = sweep._score(merged, THREE_SPEAKER_WINDOWS, speakers=3)
    assert result["pass"] is False
    assert result["per_turn"] == [0, 0, 2]
    assert result["distinct"] == 2


def test_score_fails_when_a_turn_maps_to_silence():
    missing = _timeline([(13, 25, 0), (63, 75, 2)])
    result = sweep._score(missing, THREE_SPEAKER_WINDOWS, speakers=3)
    assert result["pass"] is False
    assert result["per_turn"] == [0, None, 2]


def test_score_fails_for_a_zero_speaker_configuration_instead_of_vacuously_passing():
    # `all([])` and `0 == 0` used to make an empty sweep look like a PASS.
    result = sweep._score([], [], speakers=0)
    assert result["pass"] is False


# --------------------------------------------------------------------------- #
# window index alignment
# --------------------------------------------------------------------------- #
def test_window_indices_cover_only_the_turn_and_no_silence():
    for start, end, lo, hi in [
        (1.0, 2.0, 13, 25),
        (3.0, 4.0, 38, 50),
        (5.0, 6.0, 63, 75),
    ]:
        assert sweep._window_indices(start, end) == (lo, hi)
        window = THREE_SPEAKER_TIMELINE[lo:hi]
        assert window, f"empty window for {start}-{end}"
        assert all(label is not None for label in window), f"window {lo}:{hi} includes silence"


def test_window_indices_survive_float_noise_on_a_frame_boundary():
    # 2.24/0.08 == 28.000000000000004 and 4.48/0.08 == 56.00000000000001, i.e. the
    # turn starts exactly on a frame boundary but plain ceil() would skip a frame.
    assert 2.24 / STEP > 28
    assert sweep._window_indices(2.24, 3.24) == (28, 41)
    assert 4.48 / STEP > 56
    assert sweep._window_indices(4.48, 5.48) == (56, 69)


def test_window_indices_always_return_at_least_one_frame():
    assert sweep._window_indices(1.0, 1.02) == (13, 14)
    assert sweep._window_indices(1.0, 1.0) == (13, 14)
    assert sweep._window_indices(0.0, 0.0) == (0, 1)


def test_score_window_running_past_the_timeline_does_not_raise():
    # Slicing clamps, so an over-long window simply scores the frames that exist.
    result = sweep._score(THREE_SPEAKER_TIMELINE, [(1.0, 200.0, 0)], speakers=1)
    assert result["per_turn"] == [0]
    assert result["pass"] is True


def test_score_window_past_a_silent_tail_is_unmapped():
    result = sweep._score(THREE_SPEAKER_TIMELINE, [(86.0, 200.0, 0)], speakers=1)
    assert result["per_turn"] == [None]
    assert result["pass"] is False


def test_window_indices_match_frame_count_for_aligned_turns():
    for start in (0.56, 1.12, 2.24, 4.48, 8.96):
        lo, _hi = sweep._window_indices(start, start)
        assert lo == round(start / STEP), f"{start}s should start at frame {round(start / STEP)}"


# --------------------------------------------------------------------------- #
# _dominant_speaker: equivalence with NemoDiarService._get_dominant_speaker_id
# --------------------------------------------------------------------------- #
def test_dominant_speaker_returns_none_for_all_zero_rows():
    # Production: (0 > 0.4) == 0 everywhere, so spk_pred.sum() == 0 -> None.
    assert verify._dominant_speaker(np.zeros((6, 4))) is None


def test_dominant_speaker_ignores_sub_threshold_noise():
    # The real model never emits exact zeros in silence, so the frame mask must
    # come from the threshold - not from `max(axis=1) > 0.0`.
    noise = np.array([[0.05, 0.02, 0.0, 0.0], [0.11, 0.0, 0.03, 0.0]])
    assert verify._dominant_speaker(noise) is None
    # ...and the unthresholded rule is only reachable by asking for it explicitly.
    assert verify._dominant_speaker(noise, 0.0) == 0


def test_dominant_speaker_single_speaker_above_threshold():
    assert verify._dominant_speaker(np.array([[0.9, 0.1, 0.0, 0.0]])) == 0
    assert verify._dominant_speaker(np.array([[0.1, 0.0, 0.0, 0.95]])) == 3


def test_dominant_speaker_takes_the_majority_across_frames():
    # rows: spk0 x2, spk2 x3 -> dominant 2
    preds = np.array(
        [
            [0.9, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.1, 0.0],
            [0.0, 0.1, 0.7, 0.0],
            [0.0, 0.0, 0.6, 0.1],
            [0.1, 0.0, 0.5, 0.0],
        ]
    )
    assert verify._dominant_speaker(preds) == 2


def test_dominant_speaker_tie_matches_production_argmax():
    # 1 frame of speaker 0 and 1 frame of speaker 2: production keeps np.argmax
    # of the bincount, i.e. the lower index, not the first or the highest.
    preds = np.array([[0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0]])
    assert verify._dominant_speaker(preds) == 0


def test_dominant_speaker_argmax_runs_on_the_binarised_row_not_the_probabilities():
    # Both 0.44 and 0.45 clear the 0.4 threshold, so production's row becomes
    # [1, 0, 1, 0] and np.argmax returns the FIRST active speaker, not the
    # loudest one. The port has to keep that quirk to stay faithful.
    assert verify._dominant_speaker(np.array([[0.0, 0.0, 0.0, 0.0], [0.44, 0.0, 0.45, 0.0]])) == 0
    assert verify._dominant_speaker(np.array([[0.0, 0.0, 0.0, 0.0], [0.30, 0.0, 0.45, 0.0]])) == 2
    assert verify._dominant_speaker(np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.7]])) == 1


def test_sweep_and_verify_dominant_speaker_never_drift():
    rng = np.random.default_rng(0)
    cases = [
        np.zeros((3, 4)),
        np.array([[0.5, 0.0, 0.0, 0.0]]),
        rng.random((6, 4)) * 0.2,
        rng.random((6, 4)),
        np.array([[0.0, 0.9, 0.0, 0.0]] * 3 + [[0.0, 0.0, 0.8, 0.0]]),
    ]
    for preds in cases:
        for threshold in (0.0, 0.4, 0.5, 1.0):
            assert verify._dominant_speaker(preds, threshold) == sweep._dominant_speaker(preds, threshold)


def test_default_threshold_matches_the_production_service_default():
    # NeMoDiarInputParams.threshold is the value the bot runs with; a silent
    # drift here makes every offline verdict disagree with the endpoint.
    source = _DIAR_PY.read_text(encoding="utf-8")
    match = re.search(r"threshold:\s*Optional\[float\]\s*=\s*([0-9.]+)", source)
    assert match, "could not find NeMoDiarInputParams.threshold in diar.py"
    assert verify.DEFAULT_THRESHOLD == float(match.group(1))
    assert sweep.DEFAULT_THRESHOLD == float(match.group(1))


def test_scripts_chunk_audio_the_way_the_bot_does():
    # The offline scripts deliberately feed one 80ms chunk per diarize() call
    # (5 raw frames == 1280 samples), matching the production bot's diarization
    # cadence: the service default buffer_size is the trained 6 * 80ms = 0.48s
    # chunk, i.e. 30 raw frames of 16ms each (see NemoDiarService.__init__).
    assert verify.FRAME_LEN_SECS // verify.RAW_FRAME_SECS == 5
    assert int(verify.FRAME_LEN_SECS * 16000) == 1280 == 5 * int(verify.RAW_FRAME_SECS * 16000)
    assert int(sweep.FRAME_LEN_SECS * sweep.SAMPLE_RATE) == 1280


# --------------------------------------------------------------------------- #
# endpoint client loopback against a model-free RTVI-like server
# --------------------------------------------------------------------------- #
def _rtvi(text: str) -> dict:
    return {
        "label": "rtvi-ai",
        "type": "user-transcription",
        "id": "utterance-1",
        "data": {"text": text, "user_id": "user", "timestamp": "2026-01-01T00:00:00Z", "final": True},
    }


def _write_wav(path: Path, seconds: float = 0.1) -> Path:
    t = np.linspace(0, seconds, int(sweep.SAMPLE_RATE * seconds), endpoint=False)
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.int16), sweep.SAMPLE_RATE, subtype="PCM_16")
    return path


def _mock_server(finals: list[str], seen: list[dict]) -> object:
    """Return a websocket handler that mimics the bot's RTVI handshake."""
    serializer = ProtobufFrameSerializer()

    async def handler(ws):
        # 1. the client's client-ready, sent as a protobuf MessageFrame
        frame = await serializer.deserialize(await ws.recv())
        seen.append(frame.message)
        # 2. bot-ready, then the finals the bot would have diarized
        messages = [
            {"label": "rtvi-ai", "type": "bot-ready", "id": "bot_ready_1", "data": {"version": "1.1.0"}},
            *(_rtvi(text) for text in finals),
        ]
        for message in messages:
            await ws.send(await serializer.serialize(MessageFrame(data=json.dumps(message))))
        # 3. drain the streamed audio so the client never sees a closed socket
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            async for _message in ws:
                pass

    return handler


def _http_handler(ws_url: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            body = json.dumps({"ws_url": ws_url, "session_id": "mock"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def test_endpoint_client_reports_the_speaker_set_from_the_wire(tmp_path, capsys):
    wav = _write_wav(tmp_path / "clip.wav")
    seen: list[dict] = []

    async def drive() -> int:
        async with websockets.serve(
            _mock_server(["<speaker_1> hello there", "<speaker_2> and hello back"], seen), "127.0.0.1", 0
        ) as server:
            port = server.sockets[0].getsockname()[1]
            http = ThreadingHTTPServer(("127.0.0.1", 0), _http_handler(f"ws://127.0.0.1:{port}/connect"))
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                args = argparse.Namespace(
                    wav=wav,
                    connect_url=f"http://127.0.0.1:{http.server_address[1]}/connect",
                    expect_speakers=2,
                    drain_secs=0.5,
                    pace=True,
                )
                return await endpoint._run(args)
            finally:
                http.shutdown()
                http.server_close()

    assert asyncio.run(drive()) == 0
    output = capsys.readouterr().out

    # the handshake must be the one bridge.py:650-663 sends
    assert seen == [
        {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": seen[0]["id"],
            "data": {"version": "1.1.0", "about": {"library": "verify-diarization", "library_version": "1.0.0"}},
        }
    ]
    assert seen[0]["id"].startswith("client_ready_")
    assert "'<speaker_1> hello there'" in output
    assert "Endpoint reported 2 speaker(s): [1, 2]" in output
    assert "PASS" in output


def test_endpoint_client_fails_when_no_transcript_arrives(tmp_path, capsys):
    wav = _write_wav(tmp_path / "clip.wav")
    seen: list[dict] = []

    async def drive() -> int:
        async with websockets.serve(_mock_server([], seen), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            http = ThreadingHTTPServer(("127.0.0.1", 0), _http_handler(f"ws://127.0.0.1:{port}/connect"))
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                args = argparse.Namespace(
                    wav=wav,
                    connect_url=f"http://127.0.0.1:{http.server_address[1]}/connect",
                    expect_speakers=None,
                    drain_secs=0.3,
                    pace=False,
                )
                return await endpoint._run(args)
            finally:
                http.shutdown()
                http.server_close()

    assert asyncio.run(drive()) == 1
    output = capsys.readouterr().out
    assert "(none - is the bot running with STT + diarization enabled?)" in output
    assert "Endpoint reported 0 speaker(s): []" in output


def test_timeline_helper_frame_count_is_exact():
    # guards the fixture math the 3-speaker expectations rely on
    assert math.ceil(7.0 / STEP) == 88
    assert len(THREE_SPEAKER_TIMELINE) == 88
    assert len(_timeline([(13, 25, 0)], length=88)) == 88
