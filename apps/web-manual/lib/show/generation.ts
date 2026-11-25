"use client";

import { useEffect, useRef, useState } from "react";
import { getModel, openSession, runStep, seededRng, type ShowSession, type TextModel } from "@/lib/simulation/show";

/** Milliseconds between two tokens at normal speed. */
export const TOKEN_MS = 90;

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
