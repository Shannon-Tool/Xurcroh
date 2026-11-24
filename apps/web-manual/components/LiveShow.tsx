"use client";

import Link from "next/link";
import { useEffect, useId, useMemo, useReducer, useRef, useState } from "react";
import { Scene3DView } from "@/components/Scene3DView";
import { showCamera, showScene, workerName } from "@/lib/scene/show-scene";
import { paintBlank, paintNumbers, paintPixels, paintText } from "@/lib/show/paint";
import { ATTRACT_ORDER, beatMs, currentBeat, initialShow, showReducer, tamperRoleOf } from "@/lib/show/timeline";
import {
  SHOW_MODELS, cryptoRng, decodePixels, getModel, openSession, runStep, workerBytes,
  type Rng, type ShowKind, type ShowSession, type ShowStep,
} from "@/lib/simulation/show";

const KINDS: Array<{ kind: ShowKind; label: string }> = [
  { kind: "text", label: "Text" },
  { kind: "picture", label: "Picture" },
  { kind: "clip", label: "Clip" },
];
const ROLES = [0, 1, 2, 3];
const CLIP_FRAME_MS = 125;

/** Everything one run has produced so far. Kept outside React state because it holds large buffers. */
interface RunData {
  run: number;
  session: ShowSession;
  done: number;
  rejections: number;
  tokens: string[];
  pixels: Uint8ClampedArray | null;
  frames: Uint8ClampedArray[];
  checks: number;
  failed: number;
}

interface Snapshot {
  tokens: string[];
  frames: number;
  checks: number;
  failed: number;
  /** Bumped whenever the canvases were repainted. */
  version: number;
}

const EMPTY: Snapshot = { tokens: [], frames: 0, checks: 0, failed: 0, version: 0 };

function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * The live demonstration. A prepared answer, picture or clip is produced unit by unit; every unit is one
 * matrix product that really is computed on additive shares, checked and recombined in this page, and
 * the 3D view plays the matching steps. The models are toys and nothing here measures speed.
 */
export function LiveShow({ autoPlay = false, heading = "h2", factsOpen = false, rng }: { autoPlay?: boolean; heading?: "h1" | "h2"; factsOpen?: boolean; rng?: Rng }) {
  const titleId = useId();
  const [state, dispatch] = useReducer(showReducer, undefined, () => initialShow(ATTRACT_ORDER[0]!, { playing: false, attract: autoPlay }));
  const model = useMemo(() => getModel(state.modelId), [state.modelId]);
  const beat = currentBeat(state, model.units);
  const tamperRole = tamperRoleOf(state);

  const sectionRef = useRef<HTMLElement>(null);
  const deviceCanvas = useRef<HTMLCanvasElement>(null);
  const workerCanvases = useRef<Array<HTMLCanvasElement | null>>([null, null, null, null]);
  const filmCanvases = useRef<Array<HTMLCanvasElement | null>>([]);
  const dataRef = useRef<RunData | null>(null);
  const rngRef = useRef<Rng | null>(rng ?? null);

  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY);
  const [screens, setScreens] = useState<Record<string, HTMLCanvasElement | null>>({});
  const [onScreen, setOnScreen] = useState(true);
  const [tabVisible, setTabVisible] = useState(true);
  const [full, setFull] = useState(false);
  const [loopFrame, setLoopFrame] = useState(0);

  // The canvases double as the pictures on the 3D view's screens.
  useEffect(() => {
    setScreens({ device: deviceCanvas.current, "worker-0": workerCanvases.current[0]!, "worker-1": workerCanvases.current[1]!, "worker-2": workerCanvases.current[2]!, "worker-3": workerCanvases.current[3]! });
  }, []);

  // Start by itself only where asked, and never for readers who prefer reduced motion.
  useEffect(() => {
    if (autoPlay && !prefersReducedMotion()) dispatch({ type: "autostart" });
  }, [autoPlay]);

  // Stand still while the show is scrolled away or the tab is hidden.
  useEffect(() => {
    const onVisibility = () => setTabVisible(!document.hidden);
    document.addEventListener("visibilitychange", onVisibility);
    const section = sectionRef.current;
    const observer = section && typeof IntersectionObserver !== "undefined"
      ? new IntersectionObserver((entries) => setOnScreen(entries.some((entry) => entry.isIntersecting)))
      : null;
    if (section) observer?.observe(section);
    return () => {
      observer?.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);
  const visible = onScreen && tabVisible;

  useEffect(() => {
    if (!state.playing || !visible) return;
    const timer = window.setTimeout(() => dispatch({ type: "tick", units: model.units }), beatMs(state, model.kind, model.units));
    return () => window.clearTimeout(timer);
  }, [state, model, visible]);

  // The work itself: one protected matrix product for every unit the timeline has revealed.
  useEffect(() => {
    rngRef.current ??= cryptoRng();
    let data = dataRef.current;
    const device = deviceCanvas.current;
    if (!data || data.run !== state.run || data.session.model !== model) {
      data = { run: state.run, session: openSession(model, rngRef.current), done: 0, rejections: 0, tokens: [], pixels: null, frames: [], checks: 0, failed: 0 };
      dataRef.current = data;
      if (device) {
        if (model.kind === "text") paintText(device, model.prompt, "");
        else paintBlank(device, model.width, model.height);
      }
      for (const canvas of workerCanvases.current) if (canvas) paintBlank(canvas, 64, 40);
      for (const canvas of filmCanvases.current) if (canvas && model.kind !== "text") paintBlank(canvas, model.width, model.height);
    }
    let last: ShowStep | null = null;
    while (data.done < state.revealed && data.done < model.units) {
      last = runStep(data.session, data.done);
      data.checks += last.roles.length;
      if (last.combined && model.kind === "text") data.tokens.push(model.decodeToken(last.combined));
      if (last.combined && model.kind !== "text") {
        data.pixels = decodePixels(model, last.combined);
        if (model.kind === "clip" && data.done % 2 === 1) {
          const film = filmCanvases.current[data.frames.length];
          if (film) paintPixels(film, data.pixels, model.width, model.height);
          data.frames.push(data.pixels);
        }
      }
      data.done += 1;
    }
    if (data.rejections < state.rejections && tamperRole !== null && data.done < model.units) {
      // The reader asked for it: one worker's product is changed by one, and the real check must catch it.
      last = runStep(data.session, data.done, tamperRole);
      data.checks += last.roles.length;
      data.failed += last.roles.filter((role) => !role.valid).length;
      data.rejections = state.rejections;
    }
    if (last) {
      if (device && model.kind === "text") paintText(device, model.prompt, data.tokens.join(""));
      if (device && model.kind !== "text" && data.pixels) paintPixels(device, data.pixels, model.width, model.height);
      last.roles.forEach((role, index) => {
        const canvas = workerCanvases.current[index];
        if (!canvas) return;
        if (model.kind === "text") paintNumbers(canvas, role.product, !role.valid);
        else paintPixels(canvas, workerBytes(role.product), model.width, model.height);
      });
    }
    const produced = data;
    setSnapshot((current) => ({ tokens: [...produced.tokens], frames: produced.frames.length, checks: produced.checks, failed: produced.failed, version: current.version + 1 }));
  }, [state.run, state.revealed, state.rejections, tamperRole, model]);

  // A finished clip plays in a loop on the device's screen.
  const clipDone = model.kind === "clip" && beat === "done" && snapshot.frames === model.frames.length;
  useEffect(() => {
    if (!clipDone || prefersReducedMotion()) return;
    const timer = window.setInterval(() => setLoopFrame((current) => current + 1), CLIP_FRAME_MS);
    return () => window.clearInterval(timer);
  }, [clipDone]);
  useEffect(() => {
    const data = dataRef.current;
    const device = deviceCanvas.current;
    if (!clipDone || !data || !device || model.kind !== "clip" || data.frames.length === 0) return;
    paintPixels(device, data.frames[loopFrame % data.frames.length]!, model.width, model.height);
  }, [clipDone, loopFrame, model]);

  useEffect(() => {
    const onChange = () => {
      if (!document.fullscreenElement) setFull(false);
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  function toggleFull() {
    if (full) {
      setFull(false);
      if (document.fullscreenElement) void document.exitFullscreen?.();
      return;
    }
    setFull(true);
    void sectionRef.current?.requestFullscreen?.().catch(() => undefined);
  }

  const scene = useMemo(
    () => showScene({ kind: model.kind, beat, revealed: state.revealed, units: model.units, rejections: state.rejections, tamperRole }),
    [model, beat, state.revealed, state.rejections, tamperRole],
  );
  const direct = useMemo(() => showCamera(beat, tamperRole), [beat, tamperRole]);

  const Heading = heading;
  const FactHeading = heading === "h1" ? "h2" : "h3";
  const done = beat === "done";
  const choices = SHOW_MODELS.filter((entry) => entry.kind === model.kind);
  const progress = Math.round((100 * Math.min(state.revealed, model.units)) / model.units);
  const announcement = tamperRole !== null
    ? `${workerName(tamperRole)} failed the check. Step rejected and sent again.`
    : !done ? ""
      : model.kind === "text" ? `Answer complete: ${model.answer}`
        : model.kind === "picture" ? "Picture complete." : "Clip complete.";

  const overlay = (
    <>
      <div className="show-hud">
        <div className="show-progress" role="progressbar" aria-label="Progress of this run" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><span style={{ width: `${progress}%` }} /></div>
        <dl className="show-counters">
          <div><dt>Products checked</dt><dd data-testid="checks">{snapshot.checks}</dd></div>
          <div className={snapshot.failed > 0 ? "bad" : undefined}><dt>Rejected</dt><dd data-testid="rejected">{snapshot.failed}</dd></div>
        </dl>
      </div>
      <div className="show-side" data-inset="right">
      <div className="show-panel show-output">
        <p className="show-label">Your device</p>
        <p className="show-prompt"><span className="show-tag">Prompt</span> <q>{model.prompt}</q></p>
        {model.kind === "text" ? <p className="show-answer">{snapshot.tokens.join("")}<span className="show-caret" aria-hidden="true">▌</span></p> : null}
        <canvas ref={deviceCanvas} className="show-picture" width={96} height={60} hidden={model.kind === "text"} role="img" aria-label={model.kind === "text" ? undefined : `${model.prompt}: ${done ? "finished" : "forming, still noisy"}`} />
        {model.kind === "clip" ? (
          <ol className="show-film" aria-label={`Finished frames: ${snapshot.frames} of ${model.frames.length}`}>
            {model.frames.map((_, index) => (
              <li key={index} className={index < snapshot.frames ? "ready" : undefined}>
                <canvas ref={(element) => { filmCanvases.current[index] = element; }} width={model.width} height={model.height} aria-hidden="true" />
              </li>
            ))}
          </ol>
        ) : null}
        {tamperRole !== null ? <p className="show-rejected">{workerName(tamperRole)} failed the check. Step rejected; sending it again.</p> : null}
      </div>
      <div className="show-panel show-workers">
        <p className="show-label">What each worker sees</p>
        <ul>
          {ROLES.map((role) => (
            <li key={role} className={tamperRole === role ? "show-tile bad" : "show-tile"}>
              <canvas ref={(element) => { workerCanvases.current[role] = element; }} width={64} height={40} aria-hidden="true" />
              <span>{workerName(role)}</span>
            </li>
          ))}
        </ul>
        <p className="show-note">Real numbers from this step. On its own, each one is uniformly random.</p>
      </div>
      </div>
      <p className="show-caption">{scene.caption}</p>
    </>
  );

  return (
    <section ref={sectionRef} className={full ? "theater full" : "theater"} aria-labelledby={titleId}>
      <header className="show-head">
        <div>
          <span className="evidence">Simulation · toy models · real share arithmetic</span>
          <Heading id={titleId}>Live demonstration</Heading>
          <p>An answer, a picture and a short clip are produced below, one protected step at a time, while the 3D view shows the steps that produce them.</p>
        </div>
        <div className="show-pick">
          <div className="playback" role="group" aria-label="What to produce">
            {KINDS.map((entry) => (
              <button key={entry.kind} className="show-kind" aria-pressed={entry.kind === model.kind} onClick={() => dispatch({ type: "select", modelId: SHOW_MODELS.find((item) => item.kind === entry.kind)!.id, playing: !prefersReducedMotion() })}>{entry.label}</button>
            ))}
          </div>
          {choices.length > 1 ? (
            <div className="playback" role="group" aria-label={model.kind === "text" ? "Prepared prompts" : "Prepared pictures"}>
              {choices.map((entry) => (
                <button key={entry.id} className="show-choice" aria-pressed={entry.id === model.id} onClick={() => dispatch({ type: "select", modelId: entry.id, playing: !prefersReducedMotion() })}>{entry.title}</button>
              ))}
            </div>
          ) : null}
        </div>
      </header>

      <Scene3DView scene={scene} variant="show" direct={direct} screens={screens} screensVersion={snapshot.version + loopFrame} overlay={overlay} />

      <div className="show-controls playback" role="group" aria-label="Demonstration controls">
        <button className="show-main" onClick={() => dispatch(state.playing ? { type: "pause" } : done ? { type: "restart" } : { type: "play" })}>{state.playing ? "Pause" : "Play"}</button>
        <button onClick={() => dispatch({ type: "next", units: model.units })} disabled={done}>Next step</button>
        <button onClick={() => dispatch({ type: "restart" })}>Restart</button>
        <button onClick={() => dispatch({ type: "speed", speed: state.speed === 1 ? 2 : 1 })}>Speed: {state.speed}×</button>
        <button onClick={() => dispatch({ type: "tamper" })} disabled={done || state.tamperArmed || state.rejecting} title="Change one number in one worker's product and watch the check catch it.">Tamper with a worker</button>
        <button onClick={toggleFull}>{full ? "Exit full screen" : "Full screen"}</button>
      </div>
      <p className="sr-only" role="status">{announcement}</p>

      <p className="show-truth">
        <strong>Simulation in your browser.</strong> The answer, picture and clip are prepared toy models; the share arithmetic and its checks are real.
        Pace is set for viewing and measures nothing. Designed for defence in depth; claims under evaluation.
      </p>
      <details className="show-details" open={factsOpen}>
        <summary>What is real here</summary>
        <div className="show-facts">
          <div>
            <FactHeading>Real, computed in this page</FactHeading>
            <ul>
              <li>Every token, picture step and clip pass is one matrix product. Its input is split into two random shares, four roles each multiply one input share by one weight share, and the four products are added.</li>
              <li>Every product is tested with a Freivalds check before anything is combined. “Tamper with a worker” changes one number by one, and that check rejects it.</li>
              <li>The worker tiles and screens show the numbers each role actually holds in the current step.</li>
              <li>What you read and see is decoded from the recombined result.</li>
            </ul>
          </div>
          <div>
            <FactHeading>Prepared</FactHeading>
            <ul>
              <li>No language or image model runs. Text comes from a lookup table that holds one prepared answer per prompt.</li>
              <li>Pictures and the clip come from a small hand-built linear decoder that is driven along a prepared path from noise to the image.</li>
              <li>The four workers are four roles inside this page, not four machines. Arithmetic uses a 16-bit prime so that it stays exact at picture size; the engine works in a larger ring.</li>
              <li>The pace is chosen for viewing and measures nothing.</li>
            </ul>
          </div>
          <div>
            <FactHeading>Not shown</FactHeading>
            <ul>
              <li>Encryption in transit and anonymous routing: see the <Link href="/laboratory#onion">onion laboratory</Link>.</li>
              <li>Recovery when a worker fails: see the <Link href="/laboratory#pipeline">pipeline laboratory</Link>.</li>
              <li>Wiping memory after use: see the <Link href="/laboratory#memory">memory laboratory</Link>.</li>
              <li>Attestation, and what happens when workers pool their shares: see the <Link href="/03-threat-model">threat model</Link>.</li>
            </ul>
          </div>
        </div>
      </details>
    </section>
  );
}
