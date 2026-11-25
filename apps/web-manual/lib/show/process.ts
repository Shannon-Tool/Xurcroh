/**
 * A stage tells one computation as a row of steps. Every step carries the data as it looks once that
 * step has been done: real values, computed by the lab's engine for the question the reader chose.
 */

export type StepTone = "good" | "accent" | "cipher" | "warn" | "plain";

/** One way of showing a step's data. */
export type DataView =
  /** A list of numbers. `mark` lights the positions that matter (the words that are in the question). */
  | { kind: "numbers"; label?: string; values: Array<number | bigint | string>; show?: number; mark?: number[] }
  /** Bytes as hex, with how many there are in all and, if they are readable, what they say. */
  | { kind: "bytes"; label?: string; bytes: ArrayLike<number>; show?: number; readable?: string }
  /** One bar per value, the highest (or `top`) lit. */
  | { kind: "bars"; label?: string; values: number[]; names?: string[]; top?: number }
  /** Short lines of the form "who: what", each done, in progress, failed or still to come. */
  | { kind: "rows"; rows: Array<{ name: string; value: string; state?: "done" | "now" | "todo" | "bad" }> }
  /** One worked line of arithmetic. */
  | { kind: "sum"; text: string }
  | { kind: "text"; text: string };

export interface ProcessStep {
  id: string;
  /** Two or three words for the card. */
  title: string;
  /** One plain sentence: what this step did to the data. */
  does: string;
  data: DataView[];
  tone?: StepTone;
}

/** Digits in groups of three, so that a ten-digit number can be read. */
export function grouped(value: number | bigint | string): string {
  const text = String(value);
  if (!/^-?\d{5,}$/.test(text)) return text;
  const sign = text.startsWith("-") ? "-" : "";
  return sign + text.slice(sign.length).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

/** The first `show` values, and how many there are in all when that is more. */
export function numbersText(values: Array<number | bigint | string>, show = 4): string {
  const head = values.slice(0, show).map(grouped).join("  ");
  return values.length > show ? `${head}  … ${values.length} in all` : head;
}

export function hexText(bytes: ArrayLike<number>, show = 10): string {
  const head = Array.from({ length: Math.min(show, bytes.length) }, (_, index) => (bytes[index]! & 255).toString(16).padStart(2, "0")).join(" ");
  return bytes.length > show ? `${head} … ${bytes.length} bytes` : `${head} (${bytes.length} bytes)`;
}

/** The whole of a step's data as one line of text: what assistive technology and the tests read. */
export function viewText(view: DataView): string {
  const label = "label" in view && view.label ? `${view.label}: ` : "";
  switch (view.kind) {
    case "numbers": return label + numbersText(view.values, view.show);
    case "bytes": return label + hexText(view.bytes, view.show) + (view.readable ? ` “${view.readable}”` : "");
    case "bars": return label + view.values.map((value, index) => `${view.names?.[index] ?? index + 1}: ${grouped(value)}`).join(", ");
    case "rows": return view.rows.map((row) => `${row.name}: ${row.value}`).join("; ");
    default: return view.text;
  }
}

/** The fields a lab's story needs from its playback when one playback position is one step. */
export function stepped(playback: { index: number; speed: number; goTo(step: number): void }, last: number): { step: number; done: boolean; speed: number; onStep(step: number): void } {
  return { step: playback.index, done: playback.index >= last, speed: playback.speed, onStep: playback.goTo };
}
