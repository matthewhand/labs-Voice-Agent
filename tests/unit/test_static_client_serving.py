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

"""Unit coverage for optionally serving the built client from ``create_fastapi_app``."""

from fastapi.testclient import TestClient

from nemo_voice_agent.pipecat import bot_server


def _write_client_dist(tmp_path):
    """Create a minimal ``dist`` layout: index.html plus a static asset."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>voice</title>")
    (tmp_path / "assets" / "a.txt").write_text("asset-body")
    return tmp_path


def test_static_client_served_on_same_origin(tmp_path):
    """``/`` serves index.html, assets resolve, and /connect is not shadowed."""
    app = bot_server.create_fastapi_app(
        8765,
        public_host="voice.example",
        ws_scheme="wss",
        static_dir=str(_write_client_dist(tmp_path)),
    )
    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "voice" in root.text

        asset = client.get("/assets/a.txt")
        assert asset.status_code == 200
        assert asset.text == "asset-body"

        assert client.post("/connect").json() == {"ws_url": "wss://voice.example:8765"}


def test_no_static_dir_preserves_discovery_only_behaviour():
    """Omitting ``static_dir`` keeps today's behaviour: no UI, /connect works."""
    app = bot_server.create_fastapi_app(8765, public_host="voice.example", ws_scheme="wss")
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.post("/connect").json() == {"ws_url": "wss://voice.example:8765"}


def test_missing_static_dir_does_not_crash(tmp_path):
    """A non-existent dist dir is ignored: no mount, /connect still works."""
    app = bot_server.create_fastapi_app(
        8765,
        public_host="voice.example",
        ws_scheme="wss",
        static_dir=str(tmp_path / "not-built"),
    )
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.post("/connect").json() == {"ws_url": "wss://voice.example:8765"}


def test_public_ws_port_overrides_advertised_port():
    """A container publishing 8766->8765 advertises the host-visible port."""
    app = bot_server.create_fastapi_app(
        8765,
        public_host="10.0.0.30",
        ws_scheme="ws",
        public_ws_port=8766,
    )
    with TestClient(app) as client:
        assert client.post("/connect").json() == {"ws_url": "ws://10.0.0.30:8766"}


def test_public_ws_port_none_advertises_bound_port():
    """Leaving ``public_ws_port`` unset keeps the bound port advertised."""
    app = bot_server.create_fastapi_app(8765, public_host="10.0.0.30", ws_scheme="ws", public_ws_port=None)
    with TestClient(app) as client:
        assert client.post("/connect").json() == {"ws_url": "ws://10.0.0.30:8765"}
