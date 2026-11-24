"use client";

import { createContext, useContext, useEffect, useId, useMemo, useState } from "react";
import { QuestionPicker } from "@/components/QuestionPicker";
import type { DiagramData } from "@/lib/show/diagram-data";
import { useJourney } from "@/lib/show/question";

export type Tone = "plain" | "accent" | "warn" | "good";

/** One stop of a diagram's walk-through: which boxes and arrows are lit, and what to say about them. */
export interface DiagramStep {
  title: string;
  text: string;
  show: string[];
}

/** Makes a diagram follow the chosen question: its real values travel along it and every part can be opened. */
export interface DiagramLive {
  compute(questionId: string): DiagramData;
}

interface DiagramState {
  /** Ids lit by the current walk-through step, or null when the whole picture is shown. */
  lit: Set<string> | null;
  hover: string | null;
  setHover: (id: string | null) => void;
  /** The value on the move at the current step, when the diagram is live. */
  packet: string | null;
  /** Motion is unwanted: the value is shown standing still. */
  still: boolean;
  /** Parts that can be opened, the one that is open, and how to open one. */
  parts: Set<string>;
  picked: string | null;
  pick: (id: string) => void;
}

const DiagramContext = createContext<DiagramState>({ lit: null, hover: null, setHover: () => undefined, packet: null, still: true, parts: new Set(), picked: null, pick: () => undefined });

function pathStart(d: string): [number, number] {
  const match = /M\s*(-?[\d.]+)[\s,]+(-?[\d.]+)/.exec(d);
  return match ? [Number(match[1]), Number(match[2])] : [0, 0];
}

function stateClass(state: DiagramState, isOn: (target: Set<string> | string) => boolean): string {
  if (state.lit) return isOn(state.lit) ? " dg-on" : " dg-dim";
  if (state.hover) return isOn(state.hover) ? " dg-on" : " dg-dim";
  return "";
}

const STEP_MS = 2600;

/**
 * Frame shared by every chapter diagram: scalable, themed through CSS tokens, described for screen
 * readers. With `steps` it also offers a walk-through that lights one stage at a time.
 */
export function Diagram({ title, description, width, height, caption, steps = [], live, folded = false, children }: {
  title: string;
  description: string;
  width: number;
  height: number;
  caption: string;
  steps?: DiagramStep[];
  live?: DiagramLive;
  /** Tucks the drawing away under a "Schematic" heading, for chapters that lead with a live stage. */
  folded?: boolean;
  children: (arrowId: string) => React.ReactNode;
}) {
  const id = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const arrowId = `dg-arrow-${id}`;
  const [step, setStep] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<string | null>(null);
  const questionId = useJourney().id;
  const [picked, setPicked] = useState<string | null>(null);
  const [still, setStill] = useState(true);
  const compute = live?.compute;
  const data = useMemo(() => (compute ? compute(questionId) : null), [compute, questionId]);

  useEffect(() => {
    setStill(typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }, []);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((current) => Math.min(current === null ? 0 : current + 1, steps.length - 1));
    }, STEP_MS);
    return () => window.clearInterval(timer);
  }, [playing, steps.length]);

  // Stop at the last step instead of looping, so the final state stays readable.
  useEffect(() => {
    if (playing && step !== null && step >= steps.length - 1) setPlaying(false);
  }, [playing, step, steps.length]);

  const state = useMemo<DiagramState>(
    () => ({
      lit: step === null ? null : new Set(steps[step]?.show ?? []),
      hover,
      setHover,
      packet: data && step !== null ? data.values[step] ?? null : null,
      still,
      parts: new Set(data?.parts.map((part) => part.id) ?? []),
      picked,
      pick: (id: string) => setPicked((now) => (now === id ? null : id)),
    }),
    [steps, step, hover, data, still, picked],
  );
  const open = data?.parts.find((part) => part.id === picked) ?? null;

  const current = step === null ? null : steps[step];
  const move = (delta: number) => {
    setPlaying(false);
    setStep((value) => Math.max(0, Math.min(steps.length - 1, (value ?? -1) + delta)));
  };

  const figure = (
    <figure className="figure">
      <svg className="dg" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`dg-t-${id}`} aria-describedby={`dg-d-${id}`}>
        <title id={`dg-t-${id}`}>{title}</title>
        <desc id={`dg-d-${id}`}>{description}</desc>
        <defs>
          <marker id={arrowId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" className="dg-arrowhead" />
          </marker>
        </defs>
        <DiagramContext.Provider value={state}>{children(arrowId)}</DiagramContext.Provider>
      </svg>
      {steps.length > 0 ? (
        <div className="dg-walk">
          <div className="playback" role="group" aria-label={`Walk through: ${title}`}>
            <button onClick={() => {
              if (!playing && (step === null || step >= steps.length - 1)) setStep(0);
              setPlaying(!playing);
            }}>{playing ? "Pause" : "Play walk-through"}</button>
            <button onClick={() => move(-1)} disabled={step === null || step === 0}>Back</button>
            <button onClick={() => move(1)} disabled={step !== null && step >= steps.length - 1}>Next</button>
            <button onClick={() => { setPlaying(false); setStep(null); }} disabled={step === null}>Show all</button>
          </div>
          <p className="dg-step" aria-live="polite">
            {current
              ? <><strong>{step! + 1} of {steps.length} · {current.title}.</strong> {current.text}{data && data.values[step!] ? <> <span className="dg-value">With your question: <code>{data.values[step!]}</code></span></> : null}</>
              : <>Walk through the {steps.length} steps with Play or Next, or point at a box to see what it connects to.</>}
          </p>
          {live && data ? (
            <div className="dg-live">
              <QuestionPicker />
              <p className="dg-answer"><strong>The model&apos;s answer:</strong> {data.answer}</p>
              <div className="playback" role="group" aria-label="Look inside a part">
                <span className="dg-live-title">Look inside:</span>
                {data.parts.map((part) => (
                  <button key={part.id} aria-pressed={part.id === picked} onClick={() => state.pick(part.id)}>{part.name}</button>
                ))}
              </div>
              {open ? (
                <dl className="dg-inspect" aria-live="polite">
                  <div><dt>{open.name} holds</dt><dd>{open.holds}</dd></div>
                  <div><dt>It sees</dt><dd>{open.sees}</dd></div>
                  <div><dt>It cannot see</dt><dd>{open.cannot}</dd></div>
                </dl>
              ) : <p className="small">Choose a question, step through, and open any part to see what it holds for that question. The questions and answers are prepared; the numbers are computed in your browser.</p>}
            </div>
          ) : null}
        </div>
      ) : null}
      <figcaption className="figure-caption">{caption}</figcaption>
    </figure>
  );
  if (!folded) return figure;
  return (
    <details className="schematic">
      <summary>Schematic drawing, step by step</summary>
      {figure}
    </details>
  );
}

export function Zone({ x, y, width, height, label, tone = "plain" }: { x: number; y: number; width: number; height: number; label: string; tone?: Tone }) {
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx="12" className={`dg-zone dg-${tone}`} />
      <text x={x + 14} y={y + 24} className="dg-zone-label">{label}</text>
    </g>
  );
}

export function Box({ id, x, y, width, height, lines, tone = "plain" }: { id?: string; x: number; y: number; width: number; height: number; lines: string[]; tone?: Tone }) {
  const state = useContext(DiagramContext);
  const lineHeight = 17;
  const first = y + height / 2 - ((lines.length - 1) * lineHeight) / 2 + 5;
  const status = stateClass(state, (target) => id !== undefined && (typeof target === "string" ? target === id : target.has(id)));
  const opens = id !== undefined && state.parts.has(id);
  return (
    <g
      className={`dg-item${status}${opens ? " dg-opens" : ""}${id !== undefined && state.picked === id ? " dg-picked" : ""}`}
      onMouseEnter={id ? () => state.setHover(id) : undefined}
      onMouseLeave={id ? () => state.setHover(null) : undefined}
      onClick={opens ? () => state.pick(id) : undefined}
    >
      <rect x={x} y={y} width={width} height={height} rx="8" className={`dg-box dg-${tone}`} />
      {lines.map((line, index) => (
        <text key={line} x={x + width / 2} y={first + index * lineHeight} textAnchor="middle" className={index === 0 ? "dg-text dg-strong" : "dg-text dg-sub"}>{line}</text>
      ))}
    </g>
  );
}

export function Arrow({ id, from, to, d, arrowId, label, labelX, labelY, tone = "plain", dashed = false, both = false }: {
  id?: string;
  /** Ids of the boxes this arrow joins; pointing at either lights the arrow. */
  from?: string;
  to?: string;
  d: string;
  arrowId: string;
  label?: string;
  labelX?: number;
  labelY?: number;
  tone?: Tone;
  dashed?: boolean;
  both?: boolean;
}) {
  const state = useContext(DiagramContext);
  const status = stateClass(state, (target) => (typeof target === "string" ? target === from || target === to : id !== undefined && target.has(id)));
  // The value on the move rides the arrows of the current step.
  const packet = state.packet !== null && id !== undefined && state.lit?.has(id) === true ? state.packet : null;
  const badge = packet && packet.length > 18 ? `${packet.slice(0, 17)}…` : packet ?? "";
  const start = pathStart(d);
  return (
    <g className={`dg-item${status}`}>
      <path d={d} className={`dg-line dg-${tone}${dashed && !both ? " dg-march" : ""}`} strokeDasharray={dashed ? "6 5" : undefined} markerEnd={`url(#${arrowId})`} markerStart={both ? `url(#${arrowId})` : undefined} />
      {!dashed && !both ? <path d={d} className="dg-flow" /> : null}
      {packet ? (
        <g className="dg-packet" transform={state.still ? `translate(${start[0]} ${start[1]})` : undefined}>
          {state.still ? null : <animateMotion dur="2.6s" repeatCount="indefinite" path={d} />}
          <rect x={-(14 + badge.length * 6.6) / 2} y={-11} width={14 + badge.length * 6.6} height={22} rx={11} />
          <text y={4} textAnchor="middle">{badge}</text>
        </g>
      ) : null}
      {label && labelX !== undefined && labelY !== undefined ? <text x={labelX} y={labelY} textAnchor="middle" className="dg-label">{label}</text> : null}
    </g>
  );
}

export function Note({ x, y, lines }: { x: number; y: number; lines: string[] }) {
  return <>{lines.map((line, index) => <text key={line} x={x} y={y + index * 16} className="dg-note">{line}</text>)}</>;
}
