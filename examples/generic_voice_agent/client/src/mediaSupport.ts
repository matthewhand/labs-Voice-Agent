/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Browser capability probes for the voice client.
 *
 * A non-loopback plain-HTTP origin is not a secure context, so
 * `navigator.mediaDevices` is not exposed and microphone capture is
 * impossible. These helpers are pure (they only read globals) so they can be
 * unit-tested and reused without side effects.
 */

/** Declares the Vite env vars this client reads, without touching tsconfig. */
declare global {
  interface ImportMetaEnv {
    readonly VITE_WS_URL?: string;
    /** Vite-injected base path; '/' or '/voice/'. */
    readonly BASE_URL?: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

export interface MediaSupport {
  /** True when the page is a secure context (HTTPS or localhost). */
  secureContext: boolean;
  /** True when `navigator.mediaDevices` is exposed. */
  hasMediaDevices: boolean;
  /** Microphone capture is possible (requires both of the above). */
  micAvailable: boolean;
}

/** Microphone mode requested by the `?mic=` query parameter. */
export type MicMode = 'auto' | 'on' | 'off';

/** Detect the media capabilities of the current page. */
export function getMediaSupport(): MediaSupport {
  const secureContext =
    typeof window !== 'undefined' && window.isSecureContext === true;
  const hasMediaDevices =
    typeof navigator !== 'undefined' &&
    typeof navigator.mediaDevices !== 'undefined';
  return {
    secureContext,
    hasMediaDevices,
    micAvailable: secureContext && hasMediaDevices,
  };
}

/** Parse `?mic=0|1` (also accepts true/false/on/off). Anything else is auto. */
export function parseMicMode(search: string): MicMode {
  const value = new URLSearchParams(search).get('mic');
  if (value === '0' || value === 'false' || value === 'off') return 'off';
  if (value === '1' || value === 'true' || value === 'on') return 'on';
  return 'auto';
}

/**
 * Resolve whether the microphone should be enabled. `?mic=0` always wins;
 * otherwise the mic is only enabled when the context actually exposes it, so
 * the client degrades to receive-only instead of throwing.
 */
export function resolveMicEnabled(mode: MicMode, support: MediaSupport): boolean {
  if (mode === 'off') return false;
  return support.micAvailable;
}

/**
 * Resolve an optional server override. Precedence: `?server=<url>` query
 * parameter, then `VITE_WS_URL`. Returns `null` for the same-origin default.
 */
export function getServerOverride(
  search: string,
  envUrl?: string
): string | null {
  const fromQuery = new URLSearchParams(search).get('server');
  if (fromQuery) return fromQuery;
  if (envUrl) return envUrl;
  return null;
}
