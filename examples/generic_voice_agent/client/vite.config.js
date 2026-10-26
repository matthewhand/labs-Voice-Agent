// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

// Single source of truth for the served base path. Behind the Traefik router
// at https://<host>/voice/ the app is served under `/voice/`, otherwise `/`.
// Set VITE_BASE=/voice/ at dev/build time. Both `base` and the proxy keys
// below are derived from this value so they can never drift apart.
const base = process.env.VITE_BASE || '/';
const basePrefix = base.endsWith('/') ? base : `${base}/`;

// Backend the dev proxy forwards /connect to. Override with VITE_BOT_URL,
// e.g. VITE_BOT_URL=http://192.168.1.20:7860 npm run dev.
const botUrl = process.env.VITE_BOT_URL || 'http://127.0.0.1:7860';

// The bot's websocket server listens on its own port (WEBSOCKET_PORT, 8765 by
// default), separate from the FastAPI /connect port above. Vite proxies the
// same-origin `<base>ws` path there with `ws: true` so the client can dial
// `wss://<edge>/<base>ws` and avoid mixed-content blocking under HTTPS.
const wsTarget = process.env.VITE_WS_TARGET || 'ws://127.0.0.1:8765';

// Register each proxy under both the base-prefixed path (what the browser
// requests behind Traefik, e.g. `/voice/connect`) and the bare path (plain
// `npm run dev`). Deduplicated so base '/' yields a single entry.
const proxyKeys = (path) =>
  Array.from(new Set([`${basePrefix.replace(/\/$/, '')}${path}`, path]));

const proxy = {};
for (const key of proxyKeys('/connect')) {
  proxy[key] = { target: botUrl, changeOrigin: true };
}
for (const key of proxyKeys('/ws')) {
  proxy[key] = { target: wsTarget, changeOrigin: true, ws: true };
}

export default defineConfig({
    base: base,
    plugins: [react()],
    server: {
        host: '0.0.0.0',  // Bind to all interfaces
        port: 5173,       // Back to default Vite port
        proxy,
    },
});
