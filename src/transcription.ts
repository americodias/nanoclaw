import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import { readEnvFile } from './env.js';
import { logger } from './logger.js';

let openai: OpenAI | null = null;

function getClient(): OpenAI | null {
  if (openai) return openai;
  const env = readEnvFile(['OPENAI_API_KEY']);
  const key = env.OPENAI_API_KEY || process.env.OPENAI_API_KEY;
  if (!key) {
    logger.warn('OPENAI_API_KEY not set — voice transcription disabled');
    return null;
  }
  openai = new OpenAI({ apiKey: key });
  return openai;
}

/**
 * Transcribe an audio buffer using OpenAI Whisper.
 * Returns the transcript text, or null on failure.
 */
export async function transcribeAudio(
  audioBuffer: Buffer,
  format: string = 'ogg',
  language?: string,
): Promise<string | null> {
  const client = getClient();
  if (!client) return null;

  // Write to a temp file — OpenAI SDK needs a file-like object
  const tmpDir = path.join('/tmp', 'nanoclaw-voice');
  fs.mkdirSync(tmpDir, { recursive: true });
  const tmpFile = path.join(tmpDir, `voice-${Date.now()}.${format}`);

  try {
    fs.writeFileSync(tmpFile, audioBuffer);

    const transcription = await client.audio.transcriptions.create({
      file: fs.createReadStream(tmpFile),
      model: 'whisper-1',
      ...(language ? { language } : {}),
    });

    const text = transcription.text?.trim();
    if (text) {
      logger.info(
        { chars: text.length, format, language },
        'Transcribed voice message',
      );
    }
    return text || null;
  } catch (err) {
    logger.error({ err, format }, 'OpenAI transcription failed');
    return null;
  } finally {
    try {
      fs.unlinkSync(tmpFile);
    } catch {
      // ignore cleanup errors
    }
  }
}
