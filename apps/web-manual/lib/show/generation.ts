"use client";

import { useEffect, useRef, useState } from "react";
import { decodePixels, getModel, openSession, runStep, seededRng, type PixelModel, type ShowSession, type TextModel } from "@/lib/simulation/show";

/** Milliseconds between two tokens at normal speed. */
export const TOKEN_MS = 90;

/** How long a lab keeps its last step up: time for an answer of this many tokens to be written, and read. */
export function writingTime(tokens: number): number {
  return tokens * TOKEN_MS + 2600;
}

export interface Generation {
  /** The answer so far: every token here came out of its own protected product. */
  tokens: string[];
  total: number;
  /** Checks run and checks passed so far: four per token, one for each worker's result. */
  checks: number;
  passed: number;
  complete: boolean;
}

const EMPTY = { tokens: [] as string[], checks: 0, passed: 0 };

function reducedMotion(): boolean {
  return typeof window.matchMedia !== "function" || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Writes the chosen question's answer the way the live demonstration does: one token at a time, each
 * from one matrix product that is computed on two random pieces by four roles, checked, and put back
 * together; the token is whichever word then has the highest value. Nothing is written while `active`
 * is false, so a stage only shows an answer once the steps before it have been done.
 */
export function useGeneration(questionId: string, active: boolean, speed = 1, seed = 1): Generation {
  const [state, setState] = useState(EMPTY);
  const run = useRef<{ model: TextModel; session: ShowSession; unit: number } | null>(null);
  const model = getModel(questionId) as TextModel;

  useEffect(() => {
    if (!active) {
      run.current = null;
      setState(EMPTY);
      return;
    }
    const current = { model, session: openSession(model, seededRng(seed * 7919 + model.tokens.length)), unit: 0 };
    run.current = current;
    const produce = (upTo: number) => {
      const tokens: string[] = [];
      let passed = 0;
      for (; current.unit < upTo; current.unit += 1) {
        const step = runStep(current.session, current.unit);
        passed += step.roles.filter((role) => role.valid).length;
        if (step.combined) tokens.push(model.decodeToken(step.combined));
      }
      setState((before) => ({ tokens: [...before.tokens, ...tokens], checks: current.unit * 4, passed: before.passed + passed }));
    };
    setState(EMPTY);
    if (reducedMotion()) {
      produce(model.units);
      return;
    }
    const timer = window.setInterval(() => {
      if (current.unit >= model.units) window.clearInterval(timer);
      else produce(current.unit + 1);
    }, TOKEN_MS / speed);
    return () => window.clearInterval(timer);
  }, [model, active, speed, seed]);

  return { ...state, total: model.units, complete: active && state.tokens.length >= model.units };
}

/** Milliseconds between two steps of a picture at normal speed. */
export const PICTURE_MS = 260;

export function paintingTime(steps: number): number {
  return steps * PICTURE_MS + 2600;
}

export interface PictureGeneration {
  /** The picture as far as it has got, as RGB bytes; null before the first step. */
  pixels: Uint8ClampedArray | null;
  width: number;
  height: number;
  step: number;
  total: number;
  checks: number;
  passed: number;
  complete: boolean;
}

/**
 * Makes the chosen picture the way the live demonstration does: every step is one matrix product on
 * two random pieces, checked and put back together, and what it yields is the picture one step
 * cleaner. Nothing is drawn while `active` is false.
 */
export function usePictureGeneration(pictureId: string, active: boolean, speed = 1, seed = 1): PictureGeneration {
  const model = getModel(pictureId) as PixelModel;
  const [state, setState] = useState<{ pixels: Uint8ClampedArray | null; step: number; passed: number }>({ pixels: null, step: 0, passed: 0 });

  useEffect(() => {
    setState({ pixels: null, step: 0, passed: 0 });
    if (!active) return;
    const session = openSession(model, seededRng(seed * 104729 + model.units));
    let unit = 0;
    const produce = (upTo: number) => {
      let pixels: Uint8ClampedArray | null = null;
      let passed = 0;
      for (; unit < upTo; unit += 1) {
        const step = runStep(session, unit);
        passed += step.roles.filter((role) => role.valid).length;
        if (step.combined) pixels = decodePixels(model, step.combined);
      }
      setState((before) => ({ pixels: pixels ?? before.pixels, step: unit, passed: before.passed + passed }));
    };
    if (reducedMotion()) {
      produce(model.units);
      return;
    }
    const timer = window.setInterval(() => {
      if (unit >= model.units) window.clearInterval(timer);
      else produce(unit + 1);
    }, PICTURE_MS / speed);
    return () => window.clearInterval(timer);
  }, [model, active, speed, seed]);

  return { ...state, width: model.width, height: model.height, total: model.units, checks: state.step * 4, complete: active && state.step >= model.units };
}
