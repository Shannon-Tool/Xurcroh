export interface GifFrameInput {
  rgba: Uint8Array | Uint8ClampedArray;
  delayMs: number;
}

export interface GifWriter {
  add(rgba: Uint8Array | Uint8ClampedArray, delayMs: number): void;
  finish(): Uint8Array;
  readonly written: number;
}

export function paletteFrom(samples: ReadonlyArray<Uint8Array | Uint8ClampedArray>, colors?: number): number[][];
export function createGifWriter(options: { width: number; height: number; palette: number[][]; fuzz?: number }): GifWriter;
export function encodeGif(options: { width: number; height: number; frames: GifFrameInput[]; colors?: number; fuzz?: number }): Uint8Array;

export interface DecodedGif {
  width: number;
  height: number;
  /** 0 means it repeats for ever; null means no loop block. */
  loops: number | null;
  /** `rgba` is the whole picture as shown during that frame; `changed` counts the pixels the frame replaced. */
  frames: { delayMs: number; changed: number; rgba: Uint8Array }[];
}

export function decodeGif(bytes: Uint8Array): DecodedGif;
