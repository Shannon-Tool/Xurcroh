import type { ShowKind } from "@/lib/simulation/show";

/**
 * Pacing of the live demonstration. A run is a list of units (tokens, picture steps, clip passes).
 * The first unit is played slowly through the six steps of one protected computation; the rest
 * stream. This module only decides what is on stage when; it computes nothing itself.
 */

export type Beat = "split" | "send" | "compute" | "return" | "verify" | "combine" | "stream" | "rejected" | "done";

export const SLOW_BEATS: Beat[] = ["split", "send", "compute", "return", "verify", "combine"];
const STREAM_BEATS: Beat[] = ["stream"];

/** Order of the automatic tour that runs until the reader touches a control. */
export const ATTRACT_ORDER = ["text-shares", "picture-sunrise", "clip-sunrise", "text-tamper", "picture-city", "text-limits"];

export interface ShowState {
  modelId: string;
  /** Unit on stage, 0-based. */
  unit: number;
  /** Position within that unit's beats. */
  beat: number;
  /** How many units have had their output shown. */
  revealed: number;
  /** How many steps were rejected because a product failed its check. */
  rejections: number;
  /** A rejection is on stage; the retry follows. */
  rejecting: boolean;
  /** The reader asked for the next step to be tampered with. */
  tamperArmed: boolean;
  playing: boolean;
  speed: 1 | 2;
  /** Still on the automatic tour. */
  attract: boolean;
  /** Counts fresh starts, so restarting the same model is a visible change. */
  run: number;
  /** Whether the first unit is played slowly through all six steps. Off for the reader's own input. */
  slow: boolean;
}

export type ShowAction =
  | { type: "tick"; units: number }
  | { type: "next"; units: number }
  | { type: "autostart" }
  | { type: "play" }
  | { type: "pause" }
  | { type: "restart" }
  | { type: "tamper" }
  | { type: "speed"; speed: 1 | 2 }
  | { type: "select"; modelId: string; playing?: boolean; slow?: boolean };

export function initialShow(modelId: string, options: { playing: boolean; attract: boolean; slow?: boolean }, run = 0): ShowState {
  return { modelId, unit: 0, beat: 0, revealed: 0, rejections: 0, rejecting: false, tamperArmed: false, playing: options.playing, speed: 1, attract: options.attract, run, slow: options.slow ?? true };
}

function beatsOf(unit: number, slow: boolean): Beat[] {
  return unit === 0 && slow ? SLOW_BEATS : STREAM_BEATS;
}

export function currentBeat(state: ShowState, units: number): Beat {
  if (state.rejecting) return "rejected";
  if (state.revealed >= units) return "done";
  return beatsOf(state.unit, state.slow)[state.beat] ?? "stream";
}

/** Which of the four roles is tampered with during the rejection on stage; it moves round between rejections. */
export function tamperRoleOf(state: ShowState): number | null {
  return state.rejecting ? (state.revealed + state.rejections) % 4 : null;
}

const SLOW_MS = 1100;
const STREAM_MS: Record<ShowKind, number> = { text: 300, picture: 750, clip: 480 };
const HOLD_MS: Record<ShowKind, number> = { text: 3600, picture: 3600, clip: 6000 };
const REJECT_MS = 2400;

/** How long the current beat stays on stage. The closing hold does not shorten with speed. */
export function beatMs(state: ShowState, kind: ShowKind, units: number): number {
  const beat = currentBeat(state, units);
  if (beat === "done") return HOLD_MS[kind];
  if (beat === "rejected") return REJECT_MS / state.speed;
  return (beat === "stream" ? STREAM_MS[kind] : SLOW_MS) / state.speed;
}

/** The output of a unit appears as its last beat begins, unless the reader armed a tamper. */
function reveal(state: ShowState): ShowState {
  if (state.tamperArmed) return { ...state, tamperArmed: false, rejecting: true, rejections: state.rejections + 1 };
  return { ...state, revealed: state.revealed + 1 };
}

function enter(state: ShowState, unit: number, beat: number): ShowState {
  const placed = { ...state, unit, beat };
  return beat === beatsOf(unit, state.slow).length - 1 ? reveal(placed) : placed;
}

function advance(state: ShowState): ShowState {
  if (state.rejecting) return { ...state, rejecting: false, revealed: state.revealed + 1 };
  if (state.beat < beatsOf(state.unit, state.slow).length - 1) return enter(state, state.unit, state.beat + 1);
  return enter(state, state.unit + 1, 0);
}

export function showReducer(state: ShowState, action: ShowAction): ShowState {
  switch (action.type) {
    case "tick": {
      if (state.revealed >= action.units) {
        if (!state.attract) return { ...state, playing: false };
        const next = ATTRACT_ORDER[(ATTRACT_ORDER.indexOf(state.modelId) + 1) % ATTRACT_ORDER.length]!;
        return { ...initialShow(next, { playing: true, attract: true }, state.run + 1), speed: state.speed };
      }
      return advance(state);
    }
    case "next": {
      const paused = { ...state, playing: false, attract: false };
      if (state.revealed >= action.units) return paused;
      if (state.rejecting) return advance(paused);
      // Jump straight to the beat that shows the next unit's output.
      const unit = state.revealed;
      return reveal({ ...paused, unit, beat: beatsOf(unit, state.slow).length - 1 });
    }
    case "autostart":
      return { ...state, playing: true };
    case "play":
      return { ...state, playing: true, attract: false };
    case "pause":
      return { ...state, playing: false, attract: false };
    case "restart":
      return { ...initialShow(state.modelId, { playing: true, attract: false, slow: state.slow }, state.run + 1), speed: state.speed };
    case "tamper":
      return { ...state, tamperArmed: true, attract: false };
    case "speed":
      return { ...state, speed: action.speed, attract: false };
    case "select":
      return { ...initialShow(action.modelId, { playing: action.playing ?? true, attract: false, slow: action.slow }, state.run + 1), speed: state.speed };
  }
}
