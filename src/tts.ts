import { spawn } from 'child_process';
import { readEnvFile } from './env.js';
import { logger } from './logger.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TtsResult {
  audio: Buffer;
  format: 'ogg' | 'mp3';
  provider: string;
}

export type TtsProviderName = 'openai' | 'elevenlabs' | 'piper';

const VALID_PROVIDERS = new Set<TtsProviderName>([
  'openai',
  'elevenlabs',
  'piper',
]);
const FAILURE_COOLDOWN_MS = 300_000; // 5 minutes

const TTS_ENV_KEYS = [
  'TTS_ENABLED',
  'TTS_PROVIDER',
  'TTS_PROVIDER_CHAIN',
  'TTS_VOICE',
  'TTS_MODEL',
  'OPENAI_API_KEY',
  'ELEVENLABS_API_KEY',
  'ELEVENLABS_VOICE_ID',
  'ELEVENLABS_MODEL',
  'PIPER_HOST',
  'PIPER_PORT',
  'PIPER_VOICE',
];

// ---------------------------------------------------------------------------
// Provider state
// ---------------------------------------------------------------------------

let chain: TtsProviderName[] = [];
let currentIndex = 0;
const failedProviders: Map<string, number> = new Map(); // name → timestamp
let env: Record<string, string> = {};

function cfg(key: string, fallback = ''): string {
  return env[key] || process.env[key] || fallback;
}

// ---------------------------------------------------------------------------
// Initialisation (call once at startup from .env)
// ---------------------------------------------------------------------------

export function initTts(): boolean {
  env = readEnvFile(TTS_ENV_KEYS);

  const enabled = cfg('TTS_ENABLED', 'true') !== 'false';
  if (!enabled) {
    logger.info('TTS disabled via TTS_ENABLED=false');
    return false;
  }

  const chainStr = cfg('TTS_PROVIDER_CHAIN');
  if (chainStr) {
    chain = chainStr
      .split(',')
      .map((s) => s.trim().toLowerCase() as TtsProviderName)
      .filter((p) => VALID_PROVIDERS.has(p));
  }
  if (chain.length === 0) {
    const single = cfg('TTS_PROVIDER', 'openai')
      .trim()
      .toLowerCase() as TtsProviderName;
    if (VALID_PROVIDERS.has(single)) chain = [single];
  }
  if (chain.length === 0) {
    logger.warn('TTS: no valid providers configured');
    return false;
  }

  // Validate at least one provider has credentials
  const hasAny = chain.some((p) => providerAvailable(p));
  if (!hasAny) {
    logger.warn('TTS: no providers have required credentials');
    return false;
  }

  logger.info({ chain }, 'TTS initialised');
  return true;
}

function providerAvailable(name: TtsProviderName): boolean {
  switch (name) {
    case 'openai':
      return !!cfg('OPENAI_API_KEY');
    case 'elevenlabs':
      return !!cfg('ELEVENLABS_API_KEY');
    case 'piper':
      return true; // self-hosted, no key needed
    default:
      return false;
  }
}

export function ttsEnabled(): boolean {
  return chain.length > 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Synthesise text to audio using the provider chain with fallback.
 */
export async function synthesise(text: string): Promise<TtsResult | null> {
  if (chain.length === 0) return null;

  const cleaned = stripMarkup(text);
  if (!cleaned) return null;

  const order = getProviderOrder();
  const errors: Array<{ provider: string; error: unknown }> = [];

  for (const provider of order) {
    try {
      const result = await synthesiseWith(provider, cleaned);
      // Success — clear failure status
      failedProviders.delete(provider);
      return result;
    } catch (err) {
      logger.warn({ provider, err }, 'TTS provider failed, trying next');
      failedProviders.set(provider, Date.now());
      errors.push({ provider, error: err });
    }
  }

  logger.error(
    { errors: errors.map((e) => e.provider) },
    'All TTS providers failed',
  );
  return null;
}

/**
 * Get current provider status for diagnostics.
 */
export function getStatus(): {
  chain: string[];
  current: string;
  failed: Record<string, number>;
} {
  const now = Date.now();
  const failed: Record<string, number> = {};
  for (const [name, ts] of failedProviders) {
    const remaining = FAILURE_COOLDOWN_MS - (now - ts);
    if (remaining > 0) failed[name] = Math.ceil(remaining / 1000);
  }
  return { chain: [...chain], current: chain[currentIndex] || 'none', failed };
}

// ---------------------------------------------------------------------------
// Provider ordering with cooldown
// ---------------------------------------------------------------------------

function getProviderOrder(): TtsProviderName[] {
  const now = Date.now();
  const order: TtsProviderName[] = [];
  const cooledDown: TtsProviderName[] = [];

  for (let i = 0; i < chain.length; i++) {
    const idx = (currentIndex + i) % chain.length;
    const p = chain[idx];
    const failedAt = failedProviders.get(p);
    if (failedAt && now - failedAt < FAILURE_COOLDOWN_MS) {
      cooledDown.push(p);
    } else {
      order.push(p);
    }
  }

  // If all are in cooldown, try all anyway
  if (order.length === 0) return cooledDown;
  return order;
}

// ---------------------------------------------------------------------------
// Text preprocessing
// ---------------------------------------------------------------------------

function stripMarkup(text: string): string {
  return (
    text
      // HTML tags
      .replace(/<[^>]+>/g, '')
      // Markdown code fences
      .replace(/```[\s\S]*?```/g, '')
      // Inline code
      .replace(/`[^`]+`/g, '')
      // Bold/italic markers
      .replace(/[*_]{1,3}/g, '')
      // Collapse excessive newlines
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  );
}

function splitText(text: string, limit: number): string[] {
  if (text.length <= limit) return [text];

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > 0) {
    if (remaining.length <= limit) {
      chunks.push(remaining);
      break;
    }
    // Find sentence boundary before limit
    const slice = remaining.slice(0, limit);
    let splitAt = -1;
    for (const sep of ['. ', '! ', '? ']) {
      const idx = slice.lastIndexOf(sep);
      if (idx > splitAt) splitAt = idx + sep.length;
    }
    if (splitAt <= 0) splitAt = limit; // hard cut

    chunks.push(remaining.slice(0, splitAt).trim());
    remaining = remaining.slice(splitAt).trim();
  }
  return chunks;
}

// ---------------------------------------------------------------------------
// Provider implementations
// ---------------------------------------------------------------------------

async function synthesiseWith(
  provider: TtsProviderName,
  text: string,
): Promise<TtsResult> {
  switch (provider) {
    case 'openai':
      return synthesiseOpenai(text);
    case 'elevenlabs':
      return synthesiseElevenlabs(text);
    case 'piper':
      return synthesisePiper(text);
    default:
      throw new Error(`Unknown TTS provider: ${provider}`);
  }
}

// --- OpenAI TTS ---

async function synthesiseOpenai(text: string): Promise<TtsResult> {
  const OpenAI = (await import('openai')).default;
  const client = new OpenAI({ apiKey: cfg('OPENAI_API_KEY') });

  const voice = cfg('TTS_VOICE', 'nova') as any;
  const model = cfg('TTS_MODEL', 'tts-1');
  const chunks = splitText(text, 4096);
  const parts: Buffer[] = [];

  for (const chunk of chunks) {
    const response = await client.audio.speech.create({
      model,
      voice,
      input: chunk,
      response_format: 'opus',
    });
    parts.push(Buffer.from(await response.arrayBuffer()));
  }

  return { audio: Buffer.concat(parts), format: 'ogg', provider: 'openai' };
}

// --- ElevenLabs TTS ---

async function synthesiseElevenlabs(text: string): Promise<TtsResult> {
  const apiKey = cfg('ELEVENLABS_API_KEY');
  if (!apiKey) throw new Error('ELEVENLABS_API_KEY not set');

  const voiceId = cfg('ELEVENLABS_VOICE_ID', 'JBFqnCBsd6RMkjVDRZzb');
  const model = cfg('ELEVENLABS_MODEL', 'eleven_multilingual_v2');
  const chunks = splitText(text, 5000);
  const parts: Buffer[] = [];

  for (const chunk of chunks) {
    const res = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': apiKey,
          'Content-Type': 'application/json',
          Accept: 'audio/mpeg',
        },
        body: JSON.stringify({
          text: chunk,
          model_id: model,
          output_format: 'mp3_44100_128',
        }),
      },
    );

    if (!res.ok) {
      const body = await res.text();
      throw new Error(
        `ElevenLabs API error ${res.status}: ${body.slice(0, 200)}`,
      );
    }
    parts.push(Buffer.from(await res.arrayBuffer()));
  }

  return { audio: Buffer.concat(parts), format: 'mp3', provider: 'elevenlabs' };
}

// --- Piper TTS (via Wyoming Python helper) ---

function spawnBuffer(
  cmd: string,
  args: string[],
  input: string | Buffer,
  envVars?: Record<string, string | undefined>,
  timeoutMs = 60_000,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: envVars || process.env,
    });

    const chunks: Buffer[] = [];
    let stderr = '';
    const timer = setTimeout(() => {
      proc.kill();
      reject(new Error(`Process timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    proc.stdout.on('data', (d: Buffer) => chunks.push(d));
    proc.stderr.on('data', (d: Buffer) => {
      stderr += d.toString();
    });

    proc.on('close', (code: number) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(
          new Error(
            `Process exited with code ${code}: ${stderr.slice(0, 500)}`,
          ),
        );
      } else {
        resolve(Buffer.concat(chunks));
      }
    });
    proc.on('error', (err: Error) => {
      clearTimeout(timer);
      reject(err);
    });

    proc.stdin.write(input);
    proc.stdin.end();
  });
}

async function synthesisePiper(text: string): Promise<TtsResult> {
  const path = await import('path');

  const host = cfg('PIPER_HOST', 'localhost');
  const port = cfg('PIPER_PORT', '10200');
  const voice = cfg('PIPER_VOICE');

  const scriptDir = path.dirname(new URL(import.meta.url).pathname);
  const scriptPath = path.resolve(scriptDir, '..', 'scripts', 'piper-tts.py');
  const uvPath = (process.env.HOME || '/home/adias') + '/.local/bin/uv';
  const cortexDir = (process.env.HOME || '/home/adias') + '/cortex';
  const envVars = {
    ...process.env,
    PATH: `${process.env.HOME}/.local/bin:${process.env.PATH}`,
  };

  // Run the Python helper: stdin=text, stdout=WAV
  const args = ['run', '--project', cortexDir, scriptPath, host, port];
  if (voice) args.push(voice);

  const wavBuf = await spawnBuffer(uvPath, args, text, envVars, 30_000);
  if (!wavBuf || wavBuf.length === 0) {
    throw new Error('Piper returned no audio data');
  }

  // Convert WAV → OGG Opus via ffmpeg for Telegram voice format
  const oggBuf = await spawnBuffer(
    'ffmpeg',
    ['-i', 'pipe:0', '-c:a', 'libopus', '-b:a', '64k', '-f', 'ogg', 'pipe:1'],
    wavBuf,
    undefined,
    30_000,
  );

  return { audio: oggBuf, format: 'ogg', provider: 'piper' };
}
