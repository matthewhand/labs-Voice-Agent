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
 * Dependency-free diarization view.
 *
 * The NeMo turn-taking service prefixes the diarized speaker onto the
 * transcript text as `<speaker_N>` (see
 * `nemo_voice_agent/pipecat/services/nemo/turn_taking.py`) and emits RTVI
 * `UserTranscript` events with an empty `user_id`. This module parses that
 * tag, keeps a per-speaker roster with turn/character share, and renders one
 * transcript row per final turn. All styling is driven by CSS custom
 * properties so the existing light/dark theme vars keep working.
 */

/** A single (possibly interim) diarized transcript turn. */
export interface DiarTurn {
  /** Numeric speaker index parsed from `<speaker_N>` (0 when unknown). */
  speaker: number;
  /** Transcript text with the speaker tag stripped. */
  text: string;
  /** True for final transcripts; interim turns only drive the live indicator. */
  final: boolean;
  /** Optional time-to-first-byte in ms for this turn. */
  ttfb?: number;
}

/** Aggregate per-speaker counters used for the roster chips. */
export interface SpeakerStats {
  turns: number;
  chars: number;
}

const SPEAKER_TAG = /<speaker_(\d+)>\s*/;
const DEFAULT_STORAGE_KEY = 'diar.speakers';

/**
 * Split `<speaker_N>` off the front of a transcript and return the numeric
 * speaker index plus the clean text. Falls back to a number embedded in
 * `user_id` (currently empty on the server), then to speaker 0.
 */
export function parseSpeaker(
  text: string,
  userId?: string
): { speaker: number; text: string } {
  const raw = typeof text === 'string' ? text : '';
  const tag = SPEAKER_TAG.exec(raw);
  const clean = (tag ? raw.replace(tag[0], '') : raw).trim();

  let speaker = 0;
  if (tag) {
    speaker = Number.parseInt(tag[1], 10) || 0;
  } else if (userId) {
    const fromId = /(\d+)/.exec(userId);
    if (fromId) speaker = Number.parseInt(fromId[1], 10) || 0;
  }
  return { speaker, text: clean };
}

/**
 * Stable per-speaker hue. Golden-angle spacing keeps consecutive speaker
 * indices visually distinct without a fixed table.
 */
export function hueForSpeaker(speaker: number): number {
  return Math.round((speaker * 137.508) % 360);
}

interface DiarizationViewOptions {
  roster: HTMLElement | null;
  transcript: HTMLElement | null;
  now?: HTMLElement | null;
  storageKey?: string;
}

export class DiarizationView {
  private readonly roster: HTMLElement | null;
  private readonly transcript: HTMLElement | null;
  private readonly now: HTMLElement | null;
  private readonly storageKey: string;
  private readonly stats = new Map<number, SpeakerStats>();
  private names: Record<number, string> = {};
  private speaking: number | null = null;

  constructor(opts: DiarizationViewOptions) {
    this.roster = opts.roster;
    this.transcript = opts.transcript;
    this.now = opts.now ?? null;
    this.storageKey = opts.storageKey ?? DEFAULT_STORAGE_KEY;
    this.names = this.loadNames();
  }

  /** Human label for a speaker: the persisted rename, else `Speaker N`. */
  public label(speaker: number): string {
    return this.names[speaker] ?? `Speaker ${speaker}`;
  }

  /**
   * Feed a turn through the same render path used by live RTVI events.
   * Interim turns (`final === false`) only update the live indicator; final
   * turns append a transcript row and refresh the roster.
   */
  public addTurn(turn: DiarTurn): void {
    if (!turn.final) {
      this.setSpeaking(turn.speaker);
      return;
    }

    const parsed = parseSpeaker(turn.text, undefined);
    const clean = parsed.text.length > 0 ? parsed.text : turn.text;
    const entry = this.stats.get(turn.speaker) ?? { turns: 0, chars: 0 };
    entry.turns += 1;
    entry.chars += clean.length;
    this.stats.set(turn.speaker, entry);

    this.setSpeaking(null);
    this.appendRow(turn.speaker, clean);
    this.renderRoster();
  }

  /** Highlight (or clear) the "now speaking" indicator. */
  public setSpeaking(speaker: number | null): void {
    this.speaking = speaker;
    this.renderRoster();
    if (this.now) {
      this.now.textContent =
        speaker === null ? 'Now speaking: —' : `Now speaking: ${this.label(speaker)}`;
    }
  }

  /** Persist a rename and update every rendered label for that speaker. */
  public rename(speaker: number, name: string): void {
    const trimmed = name.trim();
    if (!trimmed) return;
    this.names[speaker] = trimmed;
    this.saveNames();
    if (this.transcript) {
      for (const row of this.transcript.querySelectorAll<HTMLElement>(
        `.diar-row[data-speaker="${speaker}"] .diar-row-speaker`
      )) {
        row.textContent = trimmed;
      }
    }
    this.renderRoster();
  }

  /** Clear the current session (transcript, counters, indicator); keep renames. */
  public reset(): void {
    this.stats.clear();
    this.speaking = null;
    if (this.transcript) this.transcript.replaceChildren();
    this.renderRoster();
    if (this.now) this.now.textContent = 'Now speaking: —';
  }

  private appendRow(speaker: number, text: string): void {
    if (!this.transcript) return;
    const row = document.createElement('div');
    row.className = 'diar-row';
    row.dataset.speaker = String(speaker);
    row.style.setProperty('--hue', String(hueForSpeaker(speaker)));

    const label = document.createElement('span');
    label.className = 'diar-row-speaker';
    label.textContent = this.label(speaker);

    const body = document.createElement('span');
    body.className = 'diar-row-text';
    body.textContent = text;

    row.append(label, body);
    this.transcript.appendChild(row);
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }

  private renderRoster(): void {
    if (!this.roster) return;
    const speakers = new Set<number>(this.stats.keys());
    if (this.speaking !== null) speakers.add(this.speaking);
    const ordered = Array.from(speakers).sort((a, b) => a - b);

    if (ordered.length === 0) {
      const empty = document.createElement('span');
      empty.className = 'diar-empty';
      empty.textContent = 'No speakers yet';
      this.roster.replaceChildren(empty);
      return;
    }

    const totalChars =
      Array.from(this.stats.values()).reduce((sum, s) => sum + s.chars, 0) || 1;
    this.roster.replaceChildren(
      ...ordered.map((speaker) => this.renderChip(speaker, totalChars))
    );
  }

  private renderChip(speaker: number, totalChars: number): HTMLElement {
    const chip = document.createElement('div');
    chip.className = 'diar-chip';
    chip.dataset.speaker = String(speaker);
    chip.style.setProperty('--hue', String(hueForSpeaker(speaker)));
    if (this.speaking === speaker) chip.classList.add('speaking');

    const name = document.createElement('button');
    name.type = 'button';
    name.className = 'diar-chip-name';
    name.textContent = this.label(speaker);
    name.title = 'Click to rename speaker';
    name.addEventListener('click', () => this.startRename(speaker, name));

    const stat = this.stats.get(speaker) ?? { turns: 0, chars: 0 };
    const share = Math.round((stat.chars / totalChars) * 100);
    const meta = document.createElement('span');
    meta.className = 'diar-chip-meta';
    meta.textContent = `${stat.turns} turns · ${share}%`;

    chip.append(name, meta);
    return chip;
  }

  /** Inline rename: swap the chip button for an input, commit on Enter/blur. */
  private startRename(speaker: number, button: HTMLButtonElement): void {
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'diar-rename';
    input.value = this.label(speaker);
    button.replaceWith(input);
    input.focus();
    input.select();

    let done = false;
    const commit = (): void => {
      if (done) return;
      done = true;
      this.rename(speaker, input.value);
      this.renderRoster();
    };
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        commit();
      } else if (event.key === 'Escape') {
        done = true;
        input.removeEventListener('blur', commit);
        this.renderRoster();
      }
    });
    input.addEventListener('blur', commit);
  }

  private loadNames(): Record<number, string> {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return {};
      const parsed = JSON.parse(raw) as Record<string, string>;
      const names: Record<number, string> = {};
      for (const [key, value] of Object.entries(parsed)) {
        const index = Number.parseInt(key, 10);
        if (!Number.isNaN(index) && typeof value === 'string') {
          names[index] = value;
        }
      }
      return names;
    } catch {
      return {};
    }
  }

  private saveNames(): void {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.names));
    } catch {
      // Persistence is best-effort; the rename still applies for this session.
    }
  }
}
