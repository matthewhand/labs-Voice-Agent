/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * RTVI Client Implementation
 *
 * This client connects to an RTVI-compatible bot server using WebSocket.
 *
 * Requirements:
 * - A running RTVI bot server reachable at same-origin `/connect` (the Vite
 *   proxy forwards it to the bot; override with `?server=` or `VITE_WS_URL`).
 */

import {
  PipecatClient,
  PipecatClientOptions,
  PipecatMetricsData,
  RTVIEvent,
  TranscriptData,
} from '@pipecat-ai/client-js';
import {
  WavMediaManager,
  WebSocketTransport
} from "@pipecat-ai/websocket-transport";
import {
  getMediaSupport,
  getServerOverride,
  MediaSupport,
  MicMode,
  parseMicMode,
  resolveMicEnabled,
} from './mediaSupport';
import { DiarizationView, parseSpeaker } from './diarization';
import type { DiarTurn } from './diarization';

/** Private fields of WavMediaManager that the receive-only subclass touches. */
interface WavMediaManagerInternals {
  _initialized?: boolean;
  _wavStreamPlayer?: {
    connect(): Promise<void>;
    interrupt(): Promise<void>;
  };
}

/**
 * Playback-only media manager for contexts where microphone capture is
 * impossible (plain-HTTP LAN origin) or was explicitly disabled.
 *
 * The default `DailyMediaManager` constructs a Daily call object in its
 * constructor, and `@daily-co/daily-js` throws "WebRTC not supported or
 * suppressed" when `navigator.mediaDevices` is missing. `WavMediaManager`
 * avoids Daily, but its device-change listeners still touch
 * `navigator.mediaDevices`; this subclass keeps only the bot-audio player and
 * skips all device enumeration/mic capture.
 */
class ReceiveOnlyMediaManager extends WavMediaManager {
  private get internals(): WavMediaManagerInternals {
    return this as unknown as WavMediaManagerInternals;
  }

  async initialize(): Promise<void> {
    if (this.internals._initialized) return;
    await this.internals._wavStreamPlayer?.connect();
    this.internals._initialized = true;
  }

  async connect(): Promise<void> {
    await this.initialize();
  }

  async disconnect(): Promise<void> {
    if (!this.internals._initialized) return;
    try {
      await this.internals._wavStreamPlayer?.interrupt();
    } catch (error) {
      console.warn('ReceiveOnlyMediaManager: error stopping playback', error);
    }
    this.internals._initialized = false;
  }
}

class WebsocketClientApp {
  private rtviClient: PipecatClient | null = null;
  private connectBtn: HTMLButtonElement | null = null;
  private disconnectBtn: HTMLButtonElement | null = null;
  private muteBtn: HTMLButtonElement | null = null;
  private resetBtn: HTMLButtonElement | null = null;
  private themeToggleBtn: HTMLButtonElement | null = null;
  private serverSelect: HTMLSelectElement | null = null;
  private statusSpan: HTMLElement | null = null;
  private statusLine: HTMLElement | null = null;
  private mediaNotice: HTMLElement | null = null;
  private debugLog: HTMLElement | null = null;
  private volumeBar: HTMLElement | null = null;
  private volumeText: HTMLElement | null = null;
  private botAudio: HTMLAudioElement;
  private isConnecting: boolean = false;
  private isDisconnecting: boolean = false;
  private isMuted: boolean = false;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private microphone: MediaStreamAudioSourceNode | null = null;
  private volumeUpdateInterval: number | null = null;
  private currentBotMessageElement: HTMLDivElement | null = null;
  private currentBotMessage: string = '';
  private currentUserMessageElement: HTMLDivElement | null = null;
  private currentUserMessage: string = '';
  /** ISO time for the active user row (set when the row is pre-allocated). */
  private currentUserUtteranceIso: string = '';

  // Media capabilities / microphone policy for this page load.
  private readonly mediaSupport: MediaSupport;
  private readonly micMode: MicMode;
  private readonly micEnabled: boolean;
  /** Optional custom server base URL from ?server= or VITE_WS_URL. */
  private readonly serverOverride: string | null;

  // Status line state (inference metrics + diarized speaker id).
  private inferenceStatus: string = '';
  private speakerStatus: string = '';

  // Per-speaker diarization view (roster + transcript) and the last TTFB seen
  // from RTVIEvent.Metrics, attached to the next transcript turn.
  private readonly diar: DiarizationView;
  private lastTtfb: number = 0;

  constructor() {
    console.log("WebsocketClientApp");
    this.mediaSupport = getMediaSupport();
    this.micMode = parseMicMode(window.location.search);
    this.micEnabled = resolveMicEnabled(this.micMode, this.mediaSupport);
    this.serverOverride = getServerOverride(
      window.location.search,
      import.meta.env?.VITE_WS_URL
    );

    this.botAudio = document.createElement('audio');
    this.botAudio.autoplay = true;
    //this.botAudio.playsInline = true;
    document.body.appendChild(this.botAudio);

    this.setupDOMElements();
    this.diar = new DiarizationView({
      roster: document.getElementById('diar-roster'),
      transcript: document.getElementById('diar-transcript'),
      now: document.getElementById('diar-now'),
    });
    this.setupServerOptions();
    this.setupEventListeners();
    this.syncThemeToggleLabel();
    this.renderMediaNotice();
    this.renderStatusLine();
    this.setupDemoHook();
  }

  /**
   * Expose a dev-only injection hook (`?demo=1`) so the diarization UI can be
   * previewed and tested without a live bot. `push` feeds the exact same
   * render path as RTVI UserTranscript events.
   */
  private setupDemoHook(): void {
    if (new URLSearchParams(window.location.search).get('demo') !== '1') return;
    window.__diarDemo = {
      push: (turn) => this.handleTranscriptTurn(turn),
      setSpeaking: (speaker) => this.diar.setSpeaking(speaker),
      reset: () => this.diar.reset(),
    };
    this.log('Demo mode enabled: window.__diarDemo available');
  }

  /**
   * Normalize a transcript turn and route it into the diarization view. Both
   * the live RTVI listener and the `?demo=1` hook call this.
   */
  private handleTranscriptTurn(turn: DiarTurn): void {
    this.diar.addTurn(turn);
    if (turn.final) {
      this.speakerStatus = `Speaker: ${this.diar.label(turn.speaker)}`;
      this.renderStatusLine();
    }
  }

  /**
   * Set up references to DOM elements and create necessary media elements
   */
  private setupDOMElements(): void {
    this.connectBtn = document.getElementById('connect-btn') as HTMLButtonElement;
    this.disconnectBtn = document.getElementById('disconnect-btn') as HTMLButtonElement;
    this.muteBtn = document.getElementById('mute-btn') as HTMLButtonElement;
    this.resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
    this.serverSelect = document.getElementById('server-select') as HTMLSelectElement;
    this.statusSpan = document.getElementById('connection-status');
    this.statusLine = document.getElementById('status-line');
    this.mediaNotice = document.getElementById('media-notice');
    this.debugLog = document.getElementById('debug-log');
    this.volumeBar = document.getElementById('volume-bar');
    this.volumeText = document.getElementById('volume-text');
    this.themeToggleBtn = document.getElementById('theme-toggle') as HTMLButtonElement;
  }

  /**
   * Set up event listeners for connect/disconnect buttons
   */
  private setupEventListeners(): void {
    this.connectBtn?.addEventListener('click', () => this.connect());
    this.disconnectBtn?.addEventListener('click', () => this.disconnect());
    this.muteBtn?.addEventListener('click', () => this.toggleMute());
    this.resetBtn?.addEventListener('click', () => this.reset());
    this.themeToggleBtn?.addEventListener('click', () => this.toggleTheme());
    this.serverSelect?.addEventListener('change', () => this.updateServerUrl());
  }

  /**
   * Add a timestamped message to the debug log
   */
  private log(message: string): void {
    if (!this.debugLog) return;
    const entry = document.createElement('div');
    entry.textContent = `${new Date().toISOString()} - ${message}`;
    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3';
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50';
    }
    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
    console.log(message);
  }

  /**
   * Create a bot message element and add it to the debug log
   */
  private createBotMessageElement(initialText: string): HTMLDivElement | null {
    if (!this.debugLog) return null;
    const entry = document.createElement('div');
    entry.style.color = '#4CAF50';
    entry.textContent = `${new Date().toISOString()} - ${initialText}`;
    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
    return entry;
  }

  /**
   * Create a user transcript row (pre-allocated on VAD user-started, filled on final transcript).
   */
  private createUserMessageElement(initialText: string): HTMLDivElement | null {
    if (!this.debugLog) return null;
    this.currentUserUtteranceIso = new Date().toISOString();
    const entry = document.createElement('div');
    entry.style.color = '#2196F3';
    entry.textContent = `${this.currentUserUtteranceIso} - User: ${initialText}`;
    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
    return entry;
  }

  /**
   * Update the connection status display
   */
  private updateStatus(status: string): void {
    if (this.statusSpan) {
      this.statusSpan.textContent = status;
    }
    this.log(`Status: ${status}`);
  }

  /**
   * Check for available media tracks and set them up if present
   * This is called when the bot is ready or when the transport state changes to ready
   */
  setupMediaTracks() {
    if (!this.rtviClient) return;
    const tracks = this.rtviClient.tracks();
    if (tracks.bot?.audio) {
      this.setupAudioTrack(tracks.bot.audio);
    }
  }

  /**
   * Set up listeners for track events (start/stop)
   * This handles new tracks being added during the session
   */
  setupTrackListeners() {
    if (!this.rtviClient) {
      this.log('Cannot setup track listeners: client is null');
      return;
    }

    try {
      // Listen for new tracks starting
      this.rtviClient.on(RTVIEvent.TrackStarted, (track, participant) => {
        // Only handle non-local (bot) tracks
        if (!participant?.local && track.kind === 'audio') {
          this.setupAudioTrack(track);
        }
      });

      // Listen for tracks stopping
      this.rtviClient.on(RTVIEvent.TrackStopped, (track, participant) => {
        this.log(`Track stopped: ${track.kind} from ${participant?.name || 'unknown'}`);
      });
    } catch (error) {
      this.log(`Error setting up track listeners: ${error}`);
    }
  }

  /**
   * Surface a minimal "inference + speakers" status line.
   *
   * RTVIEvent.Metrics carries processing/TTFB timings and RTVIEvent.UserTranscript
   * carries the transcript along with a `user_id` (the diarized speaker id when
   * the bot attributes one). If the server does not emit these events the line
   * simply stays at its placeholder.
   */
  setupStatusListeners() {
    if (!this.rtviClient) return;

    try {
      this.rtviClient.on(RTVIEvent.Metrics, (data: PipecatMetricsData) => {
        const parts: string[] = [];
        const ttfb = data?.ttfb?.[0];
        const processing = data?.processing?.[0];
        if (ttfb) {
          this.lastTtfb = Math.round(ttfb.value);
          parts.push(`${ttfb.processor} TTFB ${this.lastTtfb}ms`);
        }
        if (processing) {
          parts.push(`${processing.processor} ${Math.round(processing.value)}ms`);
        }
        if (parts.length > 0) {
          this.inferenceStatus = parts.join(', ');
          this.renderStatusLine();
          this.log(`Metrics: ${this.inferenceStatus}`);
        }
      });

      this.rtviClient.on(RTVIEvent.UserTranscript, (data: TranscriptData) => {
        // The server prefixes the diarized speaker onto the transcript text as
        // `<speaker_N>` and leaves `user_id` empty (see
        // nemo_voice_agent/pipecat/services/nemo/turn_taking.py). Parse the
        // tag and feed the same render path as the `?demo=1` hook; interim
        // turns only drive the live indicator.
        const parsed = parseSpeaker(data?.text ?? '', data?.user_id);
        this.handleTranscriptTurn({
          speaker: parsed.speaker,
          text: parsed.text,
          final: Boolean(data?.final),
          ttfb: this.lastTtfb || undefined,
        });
      });
    } catch (error) {
      this.log(`Error setting up status listeners: ${error}`);
    }
  }

  /** Clear the metrics/speaker values and refresh the status line. */
  private resetStatusLine(): void {
    this.inferenceStatus = '';
    this.speakerStatus = '';
    this.renderStatusLine();
  }

  /** Render the current inference + speaker values into the status line. */
  private renderStatusLine(): void {
    if (!this.statusLine) return;
    const inference = this.inferenceStatus || 'Inference: —';
    const speaker = this.speakerStatus || 'Speaker: —';
    this.statusLine.textContent = `${inference}  ·  ${speaker}`;
  }

  /**
   * Set up an audio track for playback
   * Handles both initial setup and track updates
   */
  private setupAudioTrack(track: MediaStreamTrack): void {
    this.log('Setting up audio track');
    if (this.botAudio.srcObject && "getAudioTracks" in this.botAudio.srcObject) {
      const oldTrack = this.botAudio.srcObject.getAudioTracks()[0];
      if (oldTrack?.id === track.id) return;
    }
    this.botAudio.srcObject = new MediaStream([track]);
  }

  /**
   * Initialize and connect to the bot
   * This sets up the RTVI client, initializes devices, and establishes the connection
   */
  public async connect(): Promise<void> {
    if (this.isConnecting) {
      this.log('Connection already in progress, ignoring...');
      return;
    }

    try {
      this.isConnecting = true;
      const startTime = Date.now();

      //const transport = new DailyTransport();
      // In receive-only mode we swap DailyMediaManager (which throws when
      // WebRTC/mediaDevices are unavailable) for a playback-only manager.
      const transport = this.micEnabled
        ? new WebSocketTransport()
        : new WebSocketTransport({ mediaManager: new ReceiveOnlyMediaManager() });
      const clientOptions: PipecatClientOptions = {
        transport,
        enableMic: this.micEnabled,
        enableCam: false,
        callbacks: {
          onConnected: () => {
            this.updateStatus('Connected');
            if (this.connectBtn) this.connectBtn.disabled = true;
            if (this.disconnectBtn) this.disconnectBtn.disabled = false;
            if (this.muteBtn) {
              this.muteBtn.disabled = !this.micEnabled;
              this.muteBtn.textContent = 'Mute';
            }
            if (this.resetBtn) this.resetBtn.disabled = false;
            if (this.serverSelect) this.serverSelect.disabled = true;
            // Start volume monitoring when connected (mic input only)
            if (this.micEnabled && !this.isMuted) {
              this.startVolumeMonitoring();
            }
          },
          onDisconnected: () => {
            // Only handle disconnect if we're not in the middle of error cleanup
            if (!this.isConnecting) {
              this.updateStatus('Disconnected');
              if (this.connectBtn) this.connectBtn.disabled = false;
              if (this.disconnectBtn) this.disconnectBtn.disabled = true;
              if (this.muteBtn) {
                this.muteBtn.disabled = true;
                this.muteBtn.textContent = 'Mute';
              }
              if (this.resetBtn) this.resetBtn.disabled = true;
              if (this.serverSelect) this.serverSelect.disabled = false;
              // Stop volume monitoring when disconnected
              this.stopVolumeMonitoring();
              this.log('Client disconnected');
            }
          },
          onBotReady: (data) => {
            this.log(`Bot ready: ${JSON.stringify(data)}`);
            this.setupMediaTracks();
          },
          onUserStartedSpeaking: () => {
            if (this.currentUserMessage !== '') {
              this.currentUserMessage = '';
              this.currentUserMessageElement = this.createUserMessageElement('');
            } else if (!this.currentUserMessageElement) {
              this.currentUserMessage = '';
              this.currentUserMessageElement = this.createUserMessageElement('');
            }
          },
          onUserTranscript: (data) => {
            if (!data.final) {
              return;
            }
            if (!this.currentUserMessageElement) {
              this.currentUserMessage = '';
              this.currentUserMessageElement = this.createUserMessageElement('');
            }
            if (this.currentUserMessageElement) {
              this.currentUserMessageElement.textContent = `${this.currentUserUtteranceIso} - User: ${data.text}`;
            }
            this.currentUserMessage = data.text;
            this.debugLog?.scrollTo({ top: this.debugLog.scrollHeight, behavior: 'smooth' });
            console.log(`User: ${data.text}`);
          },
          onBotTranscript: (data) => {
            // If no current element exists, create one (fallback in case BOT_LLM_STARTED didn't fire)
            if (!this.currentBotMessageElement) {
              this.currentBotMessage = '';
              this.currentBotMessageElement = this.createBotMessageElement('Bot: ');
            }
            
            // Accumulate the text
            this.currentBotMessage += data.text;
            
            // Update the current element
            if (this.currentBotMessageElement) {
              const timestamp = new Date().toISOString();
              this.currentBotMessageElement.textContent = `${timestamp} - Bot: ${this.currentBotMessage}`;
              this.debugLog?.scrollTo({ top: this.debugLog.scrollHeight, behavior: 'smooth' });
            }
          },
          onBotLlmStarted: () => {
            // Only create a new bot message element if the current one has content
            if (this.currentBotMessage !== '') {
              this.currentBotMessage = '';
              this.currentBotMessageElement = this.createBotMessageElement('Bot: ');
            } else if (!this.currentBotMessageElement) {
              // Create element if it doesn't exist at all
              this.currentBotMessage = '';
              this.currentBotMessageElement = this.createBotMessageElement('Bot: ');
            }
          },
          onMessageError: (error) => console.error('Message error:', error),
          onError: (error) => console.error('Error:', error),
        },
      }
      
      // Create the client with error handling
      try {
        this.rtviClient = new PipecatClient(clientOptions);
        this.setupTrackListeners();
        this.setupStatusListeners();
        this.resetStatusLine();
      } catch (clientError) {
        this.log(`Error creating RTVI client: ${clientError}`);
        throw clientError;
      }

      if (this.micEnabled) {
        this.log('Initializing devices...');
        await this.rtviClient.initDevices();
        this.log('Devices initialized successfully');
      } else {
        // Microphone state is uninitialized, so PipecatClient.connect() would
        // otherwise call initDevices() for us. In receive-only mode there are
        // no devices to initialize and the transport is playback-only.
        this.log('Receive-only mode: skipping microphone initialization');
      }

      const endpoint = this.resolveEndpoint();
      this.log(`Connecting to bot at ${endpoint}...`);
      // Connection params moved from the constructor to connect() in
      // client-js 1.x.
      if (this.isCustomServer()) {
        // Explicit override: trust the server's own ws URL untouched.
        await this.rtviClient.connect({ endpoint });
      } else {
        // Same-origin: discover via the edge proxy, then rewrite the absolute
        // ws:// URL the bot returns into a same-origin path so an HTTPS page
        // is not blocked by mixed content.
        const discovery = (await this.rtviClient.startBot({ endpoint })) as
          | { ws_url?: string; wsUrl?: string }
          | null;
        const raw = discovery?.ws_url ?? discovery?.wsUrl ?? '';
        const wsUrl = this.resolveSameOriginWs(raw);
        this.log(`Bot websocket: ${raw || '(none)'} -> ${wsUrl}`);
        await this.rtviClient.connect({ wsUrl });
      }

      const timeTaken = Date.now() - startTime;
      this.log(`Connection complete, timeTaken: ${timeTaken}`);
    } catch (error) {
      this.log(`Error connecting: ${(error as Error).message}`);
      this.updateStatus('Error');
      // Clean up if there's an error
      await this.cleanupOnError();
    } finally {
      this.isConnecting = false;
    }
  }

  /**
   * Clean up resources when there's an error during connection
   */
  private async cleanupOnError(): Promise<void> {
    // Set disconnecting flag to prevent onDisconnected callback interference
    this.isDisconnecting = true;
    
    // Store reference to client before it might become null
    const client = this.rtviClient;
    
    if (client) {
      try {
        // Check if the client is in a state where disconnect can be called
        if (typeof client.disconnect === 'function') {
          await client.disconnect();
        }
      } catch (disconnectError) {
        this.log(`Error during cleanup disconnect: ${disconnectError}`);
      } finally {
        // Always reset the client to null to allow reconnection
        this.rtviClient = null;
      }
    } else {
      this.log('Client was already null during cleanup');
    }
    
    // Reset button states
    if (this.connectBtn) this.connectBtn.disabled = false;
    if (this.disconnectBtn) this.disconnectBtn.disabled = true;
    if (this.muteBtn) {
      this.muteBtn.disabled = true;
      this.muteBtn.textContent = 'Mute';
    }
    if (this.resetBtn) this.resetBtn.disabled = true;
    if (this.serverSelect) this.serverSelect.disabled = false;
    
    // Stop volume monitoring
    this.stopVolumeMonitoring();
    
    // Clean up bot message state
    this.currentBotMessage = '';
    this.currentBotMessageElement = null;
    this.currentUserMessage = '';
    this.currentUserMessageElement = null;
    this.currentUserUtteranceIso = '';

    // Reset mute state
    this.isMuted = false;

    // Reset disconnect/status state
    this.resetStatusLine();
    this.isDisconnecting = false;
  }

  /**
   * Disconnect from the bot and clean up media resources
   */
  public async disconnect(): Promise<void> {
    if (this.isDisconnecting) {
      this.log('Disconnection already in progress, ignoring...');
      return;
    }

    this.isDisconnecting = true;
    
    // Store reference to client before it might become null
    const client = this.rtviClient;
    
    if (client) {
      try {
        // Check if the client is in a state where disconnect can be called
        if (typeof client.disconnect === 'function') {
          await client.disconnect();
        }
      } catch (error) {
        this.log(`Error disconnecting: ${(error as Error).message}`);
      } finally {
        // Always clean up resources and reset the client
        this.rtviClient = null;
        if (this.botAudio.srcObject && "getAudioTracks" in this.botAudio.srcObject) {
          this.botAudio.srcObject.getAudioTracks().forEach((track) => track.stop());
          this.botAudio.srcObject = null;
        }
      }
    } else {
      this.log('Client was already null during disconnect');
    }
    
    // Stop volume monitoring
    this.stopVolumeMonitoring();
    
    // Clean up bot message state
    this.currentBotMessage = '';
    this.currentBotMessageElement = null;
    this.currentUserMessage = '';
    this.currentUserMessageElement = null;
    this.currentUserUtteranceIso = '';

    // Reset mute state
    this.isMuted = false;

    // Reset status state
    this.resetStatusLine();

    this.isDisconnecting = false;
  }

  /**
   * Toggle microphone mute/unmute
   */
  private toggleMute(): void {
    if (!this.rtviClient) {
      this.log('Cannot toggle mute: client is null');
      return;
    }
    if (!this.micEnabled) {
      this.log('Cannot toggle mute: microphone is disabled in receive-only mode');
      return;
    }

    this.isMuted = !this.isMuted;
    this.rtviClient.enableMic(!this.isMuted);
    
    // Update button text
    if (this.muteBtn) {
      this.muteBtn.textContent = this.isMuted ? 'Unmute' : 'Mute';
    }
    
    // Update volume monitoring
    if (this.isMuted) {
      this.stopVolumeMonitoring();
    } else {
      this.startVolumeMonitoring();
    }
    
    this.log(this.isMuted ? 'Microphone muted' : 'Microphone unmuted');
  }

  /**
   * Start monitoring microphone volume
   */
  private async startVolumeMonitoring(): Promise<void> {
    try {
      if (!this.audioContext) {
        this.audioContext = new AudioContext();
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;
      
      this.microphone = this.audioContext.createMediaStreamSource(stream);
      this.microphone.connect(this.analyser);
      
      // Start continuous volume updates
      this.volumeUpdateInterval = window.setInterval(() => {
        this.updateVolumeDisplay();
      }, 100); // Update every 100ms
      
      this.log('Volume monitoring started');
    } catch (error) {
      this.log(`Error starting volume monitoring: ${error}`);
    }
  }

  /**
   * Stop monitoring microphone volume
   */
  private stopVolumeMonitoring(): void {
    if (this.volumeUpdateInterval) {
      clearInterval(this.volumeUpdateInterval);
      this.volumeUpdateInterval = null;
    }
    
    if (this.microphone) {
      this.microphone.disconnect();
      this.microphone = null;
    }
    
    // Reset volume display
    this.updateVolumeDisplay(0);
    this.log('Volume monitoring stopped');
  }

  /**
   * Update the volume display
   */
  private updateVolumeDisplay(volume?: number): void {
    if (!this.volumeBar || !this.volumeText) return;

    if (volume === undefined && this.analyser) {
      const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      this.analyser.getByteFrequencyData(dataArray);
      
      // Calculate average volume
      const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
      volume = (average / 255) * 100;
    }

    const displayVolume = volume || 0;
    const clampedVolume = Math.min(100, Math.max(0, displayVolume));
    
    this.volumeBar.style.width = `${clampedVolume}%`;
    this.volumeText.textContent = `${Math.round(clampedVolume)}%`;
    
    // Update color based on volume level
    if (clampedVolume < 30) {
      this.volumeBar.style.background = '#4caf50'; // Green
    } else if (clampedVolume < 70) {
      this.volumeBar.style.background = '#ff9800'; // Orange
    } else {
      this.volumeBar.style.background = '#f44336'; // Red
    }
  }

  /**
   * Reset the conversation context by calling the server action
   */
  private async reset(): Promise<void> {
    // Clear the local diarization view too, so the transcript pane matches the
    // server-side context that is about to be dropped.
    this.diar.reset();
    if (!this.rtviClient) {
      this.log('Cannot reset: not connected to server');
      return;
    }

    try {
      this.log('Resetting conversation context...');
      
      // `action()` was removed in client-js 1.x; custom server calls now go
      // through sendClientRequest(), which maps to the bot's on_client_message
      // handler and resolves with the server-response payload.
      const result = await this.rtviClient.sendClientRequest('reset', {});

      if (result) {
        this.log('Conversation context reset successfully');
      } else {
        this.log('Failed to reset conversation context');
      }
    } catch (error) {
      this.log(`Error resetting context: ${error}`);
    }
  }

  /**
   * Reflect the active theme on the toggle button (shows the theme it switches to).
   */
  private syncThemeToggleLabel(): void {
    if (!this.themeToggleBtn) return;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    this.themeToggleBtn.textContent = isDark ? 'Light' : 'Dark';
  }

  /**
   * Apply a theme, persist the choice, and refresh the toggle label.
   */
  private applyTheme(theme: 'light' | 'dark'): void {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('theme', theme);
    } catch (error) {
      this.log(`Unable to persist theme preference: ${error}`);
    }
    this.syncThemeToggleLabel();
  }

  /**
   * Toggle between the light and dark themes.
   */
  private toggleTheme(): void {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    this.applyTheme(isDark ? 'light' : 'dark');
  }

  /**
   * Configure the server-selection dropdown from the resolved override.
   * The default is always same-origin `/connect`, which the Vite dev proxy
   * forwards to the bot; the custom option is only selectable when a
   * `?server=` or `VITE_WS_URL` override exists.
   */
  private setupServerOptions(): void {
    if (!this.serverSelect) return;
    const customOption = this.serverSelect.querySelector(
      'option[value="custom"]'
    ) as HTMLOptionElement | null;
    if (customOption) {
      if (this.serverOverride) {
        customOption.disabled = false;
        customOption.textContent = `Custom (${this.serverOverride})`;
      } else {
        customOption.disabled = true;
        customOption.textContent = 'Custom server (not configured)';
      }
    }
    this.serverSelect.value = 'same-origin';
  }

  /** True when the user selected an explicit `?server=` / `VITE_WS_URL` server. */
  private isCustomServer(): boolean {
    return this.serverSelect?.value === 'custom' && this.serverOverride !== null;
  }

  /** The Vite base path, normalized with a trailing slash ('/' or '/voice/'). */
  private basePath(): string {
    const base = import.meta.env?.BASE_URL || '/';
    return base.endsWith('/') ? base : `${base}/`;
  }

  /**
   * Resolve the bot discovery endpoint. Same-origin `<base>connect` by default
   * (handled by the Vite/edge proxy; `/voice/connect` under base `/voice/`),
   * with an optional `?server=` / `VITE_WS_URL` override.
   */
  private resolveEndpoint(): string {
    if (this.isCustomServer() && this.serverOverride) {
      return `${this.serverOverride.replace(/\/+$/, '')}/connect`;
    }
    return `${this.basePath()}connect`;
  }

  /**
   * Rewrite the bot's absolute `ws://host:port` websocket URL to the
   * same-origin `<base>ws` path (ws/wss matching the page protocol).
   *
   * The Vite dev server proxies `<base>ws` to the bot's websocket port with
   * `ws: true` (see vite.config.js); that local hop was verified. Forwarding
   * `<base>ws` through a production edge proxy is NOT verified here, so treat
   * it as an assumption: if the edge does not pass the upgrade through, `wss`
   * under HTTPS will fail while the receive-only UI keeps working. An
   * already-same-origin URL is kept as-is.
   */
  private resolveSameOriginWs(raw: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const sameOrigin = `${protocol}//${window.location.host}${this.basePath()}ws`;
    if (!raw) return sameOrigin;
    try {
      const url = new URL(raw);
      if (url.host === window.location.host && url.protocol === protocol) {
        return raw;
      }
    } catch {
      // Not an absolute URL; fall through to the same-origin path.
    }
    return sameOrigin;
  }

  private updateServerUrl(): void {
    this.log(`Server changed to: ${this.resolveEndpoint()}`);

    // If connected, show a message that they need to reconnect
    if (this.rtviClient) {
      this.log('Please disconnect and reconnect to use the new server');
    }
  }

  /**
   * Explain the active microphone policy in the UI. On a non-secure context
   * the client runs receive-only, which is easy to mistake for a bug, so the
   * notice spells out the cause and both workarounds.
   */
  private renderMediaNotice(): void {
    if (!this.mediaNotice) return;

    let message = '';
    if (!this.mediaSupport.secureContext || !this.mediaSupport.hasMediaDevices) {
      message =
        'Microphone input is unavailable because this page is not a secure ' +
        'context (plain HTTP on a LAN address). Running in receive-only mode: ' +
        'you can hear the bot but cannot speak to it. To enable the mic, open ' +
        'the page over HTTPS or via localhost, or allow this origin under ' +
        'chrome://flags/#unsafely-treat-insecure-origin-as-secure and restart Chrome.';
    } else if (this.micMode === 'on' && !this.micEnabled) {
      message =
        'Microphone was requested with ?mic=1 but is unavailable in this ' +
        'context. Running in receive-only mode.';
    } else if (this.micMode === 'off') {
      message =
        'Microphone disabled with ?mic=0. Running in receive-only mode; ' +
        'bot audio and transcripts remain available.';
    }

    if (!message) {
      this.mediaNotice.hidden = true;
      this.mediaNotice.textContent = '';
      return;
    }

    this.mediaNotice.textContent = message;
    this.mediaNotice.hidden = false;
    this.log(`Media notice: ${message}`);
  }
}

declare global {
  interface Window {
    WebsocketClientApp: typeof WebsocketClientApp;
    /** Dev-only (`?demo=1`) diarization injection hook. */
    __diarDemo?: {
      push: (turn: DiarTurn) => void;
      setSpeaking: (speaker: number | null) => void;
      reset: () => void;
    };
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.WebsocketClientApp = WebsocketClientApp;
  new WebsocketClientApp();
});
