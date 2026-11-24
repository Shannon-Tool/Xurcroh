"use client";

import Link from "next/link";
import { useEffect, useId, useMemo, useReducer, useRef, useState, useSyncExternalStore } from "react";
import { Scene3DView } from "@/components/Scene3DView";
import { wordsSetting } from "@/lib/scene/settings";
import { showCamera, showScene, showWhy, workerName } from "@/lib/scene/show-scene";
import { inkToRgb, presetInk, stampInk, type PadPreset } from "@/lib/show/pad";
import { paintBlank, paintNumbers, paintPixels, paintText } from "@/lib/show/paint";
import { ATTRACT_ORDER, beatMs, currentBeat, initialShow, showReducer, tamperRoleOf } from "@/lib/show/timeline";
import {
  PAD, SHOW_MODELS, SHOW_PRIME, cryptoRng, decodePixels, getModel, openSession, ownDrawingModel, ownTextModel, pooledView, runStep,
  sealForWire, workerBytes, type Rng, type ShowSession, type ShowStep,
} from "@/lib/simulation/show";

type Mode = "text" | "picture" | "clip" | "own-text" | "own-drawing";
const MODES: Array<{ mode: Mode; label: string }> = [
  { mode: "text", label: "Text" },
  { mode: "picture", label: "Picture" },
  { mode: "clip", label: "Clip" },
  { mode: "own-text", label: "Your text" },
  { mode: "own-drawing", label: "Your drawing" },
];

type Eye = "you" | "worker" | "same" | "match" | "wire";
const EYES: Array<{ eye: Eye; label: string; title: string }> = [
  { eye: "you", label: "You", title: "Your device" },
  { eye: "worker", label: "One worker", title: `${workerName(0)} alone` },
  { eye: "same", label: "Two workers, same piece", title: `${workerName(0)} and ${workerName(1)} together` },
  { eye: "match", label: "Two workers, matching pieces", title: `${workerName(0)} and ${workerName(3)} together` },
  { eye: "wire", label: "The network", title: "Someone watching the network" },
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
  /** Running sum of the parts, for models whose output arrives in parts. */
  total: Int32Array | null;
  frames: Uint8ClampedArray[];
  checks: number;
  failed: number;
  last: ShowStep | null;
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

/** What another party gets to look at: either raw numbers (as text) or, when it can rebuild it, the real output. */
interface EyeView {
  text: string;
  real: boolean;
  bytes: number;
}

function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function hex(values: ArrayLike<number>, digits: number, count: number): string {
  return Array.from({ length: Math.min(count, values.length) }, (_, index) => values[index]!.toString(16).padStart(digits, "0")).join(" ");
}

/**
 * The live demonstration. An answer, picture or clip is produced unit by unit; every unit is one matrix
 * product that really is computed on additive shares, checked and recombined in this page, and the 3D
 * view plays the matching steps. The models are toys and nothing here measures speed.
 */
export function LiveShow({ autoPlay = false, heading = "h2", factsOpen = false, compact = false, rng }: {
  autoPlay?: boolean;
  heading?: "h1" | "h2";
  factsOpen?: boolean;
  /** A smaller version for use inside a chapter: it starts on the reader's own text. */
  compact?: boolean;
  rng?: Rng;
}) {
  const titleId = useId();
  const [state, dispatch] = useReducer(showReducer, undefined, () => initialShow(compact ? "own-text" : ATTRACT_ORDER[0]!, { playing: false, attract: autoPlay && !compact }));
  const [ownText, setOwnText] = useState("hello world");
  const [ownInk, setOwnInk] = useState(() => presetInk("smile"));
  const model = useMemo(
    () => (state.modelId === "own-text" ? ownTextModel(ownText) : state.modelId === "own-drawing" ? ownDrawingModel(ownInk) : getModel(state.modelId)),
    [state.modelId, ownText, ownInk],
  );
  const mode: Mode = model.id === "own-text" || model.id === "own-drawing" ? model.id : model.kind;
  const beat = currentBeat(state, model.units);
  const tamperRole = tamperRoleOf(state);
  const words = useSyncExternalStore(wordsSetting.subscribe, wordsSetting.get, () => "plain" as const);

  const sectionRef = useRef<HTMLElement>(null);
  const deviceCanvas = useRef<HTMLCanvasElement>(null);
  const eyeCanvas = useRef<HTMLCanvasElement>(null);
  const padCanvas = useRef<HTMLCanvasElement>(null);
  const workerCanvases = useRef<Array<HTMLCanvasElement | null>>([null, null, null, null]);
  const filmCanvases = useRef<Array<HTMLCanvasElement | null>>([]);
  const dataRef = useRef<RunData | null>(null);
  const rngRef = useRef<Rng | null>(rng ?? null);
  const inkRef = useRef<Uint8ClampedArray>(ownInk);
  const drawingRef = useRef(false);

  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY);
  const [screens, setScreens] = useState<Record<string, HTMLCanvasElement | null>>({});
  const [onScreen, setOnScreen] = useState(true);
  const [tabVisible, setTabVisible] = useState(true);
  const [full, setFull] = useState(false);
  const [loopFrame, setLoopFrame] = useState(0);
  const [eye, setEye] = useState<Eye>("you");
  const [eyeView, setEyeView] = useState<EyeView>({ text: "", real: false, bytes: 0 });

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
      data = { run: state.run, session: openSession(model, rngRef.current), done: 0, rejections: 0, tokens: [], pixels: null, total: null, frames: [], checks: 0, failed: 0, last: null };
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
        let whole = last.combined;
        if (model.accumulate) {
          // The output arrives in parts; the device adds the verified parts up.
          const total = data.total ?? new Int32Array(whole.length);
          for (let row = 0; row < total.length; row += 1) total[row] = (total[row]! + whole[row]!) % SHOW_PRIME;
          data.total = total;
          whole = total;
        }
        data.pixels = decodePixels(model, whole);
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
      data.last = last;
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

  // Through someone else's eyes: rebuilt from what that party really holds in the latest step.
  useEffect(() => {
    if (eye === "you") return;
    const data = dataRef.current;
    const canvas = eyeCanvas.current;
    const last = data?.last;
    if (!data || !last) {
      setEyeView({ text: "", real: false, bytes: 0 });
      if (canvas && model.kind !== "text") paintBlank(canvas, model.width, model.height);
      return;
    }
    let current = true;
    const showNumbers = (values: ArrayLike<number>, digits: number, bytes: number) => {
      if (!current) return;
      if (canvas && model.kind !== "text") {
        const pixels = new Uint8ClampedArray(model.width * model.height * 3);
        for (let index = 0; index < pixels.length; index += 1) pixels[index] = values[index % values.length]! & 255;
        paintPixels(canvas, pixels, model.width, model.height);
      }
      setEyeView({ text: hex(values, digits, 28), real: false, bytes });
    };
    if (eye === "worker") {
      showNumbers(last.roles[0]!.product, 4, 0);
    } else if (eye === "same") {
      showNumbers(pooledView(data.session, last, 0, 1).sum, 4, 0);
    } else if (eye === "wire") {
      void sealForWire(last.roles[0]!.product).then((sealed) => showNumbers(sealed, 2, sealed.length));
    } else {
      // Matching pieces: the pair rebuilds the output. Confirm it equals what the device combined.
      const pooled = pooledView(data.session, last, 0, 3);
      const same = pooled.output !== null && (last.combined === null || pooled.output.every((value, index) => value === last.combined![index]));
      if (!same) {
        showNumbers(pooled.sum, 4, 0);
      } else {
        if (canvas && model.kind !== "text" && data.pixels) paintPixels(canvas, data.pixels, model.width, model.height);
        setEyeView({ text: "", real: true, bytes: 0 });
      }
    }
    return () => {
      current = false;
    };
  }, [eye, snapshot.version, model]);

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

  // The drawing pad shows the ink as it is, including a stroke still in progress.
  useEffect(() => {
    inkRef.current = new Uint8ClampedArray(ownInk);
    if (padCanvas.current) paintPixels(padCanvas.current, inkToRgb(ownInk), PAD.width, PAD.height);
  }, [ownInk, mode]);

  function toggleFull() {
    if (full) {
      setFull(false);
      if (document.fullscreenElement) void document.exitFullscreen?.();
      return;
    }
    setFull(true);
    void sectionRef.current?.requestFullscreen?.().catch(() => undefined);
  }

  function selectMode(next: Mode) {
    if (next === "own-text" || next === "own-drawing") dispatch({ type: "select", modelId: next, slow: false });
    else dispatch({ type: "select", modelId: SHOW_MODELS.find((item) => item.kind === next)!.id, playing: !prefersReducedMotion() });
  }

  function typeOwnText(value: string) {
    setOwnText(value);
    dispatch({ type: "select", modelId: "own-text", slow: false });
  }

  function commitInk(ink: Uint8ClampedArray) {
    setOwnInk(new Uint8ClampedArray(ink));
    dispatch({ type: "select", modelId: "own-drawing", slow: false });
  }

  function dab(event: React.PointerEvent<HTMLCanvasElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return;
    const x = Math.floor(((event.clientX - box.left) / box.width) * PAD.width);
    const y = Math.floor(((event.clientY - box.top) / box.height) * PAD.height);
    if (stampInk(inkRef.current, x, y)) paintPixels(event.currentTarget, inkToRgb(inkRef.current), PAD.width, PAD.height);
  }

  const scene = useMemo(
    () => showScene({ kind: model.kind, beat, revealed: state.revealed, units: model.units, rejections: state.rejections, tamperRole, words }),
    [model, beat, state.revealed, state.rejections, tamperRole, words],
  );
  const direct = useMemo(() => showCamera(beat, tamperRole), [beat, tamperRole]);

  const Heading = heading;
  const FactHeading = heading === "h1" ? "h2" : "h3";
  const done = beat === "done";
  const choices = SHOW_MODELS.filter((entry) => entry.kind === model.kind && mode === model.kind);
  const progress = Math.round((100 * Math.min(state.revealed, model.units)) / model.units);
  const announcement = tamperRole !== null
    ? `${workerName(tamperRole)} failed the check. Step rejected and sent again.`
    : !done ? ""
      : model.kind === "text" ? `Answer complete: ${model.answer}`
        : model.kind === "picture" ? "Picture complete." : "Clip complete.";

  const eyeEntry = EYES.find((entry) => entry.eye === eye)!;
  const eyeNote = eye === "worker" ? `${workerName(0)} holds one piece of your data and one piece of the model. This is everything it has.`
    : eye === "same" ? `${workerName(0)} and ${workerName(1)} hold the same piece of your data. Together they still have only noise.`
      : eye === "wire" ? `Someone watching the network sees ${eyeView.bytes} encrypted bytes (AES-GCM here). They learn how much was sent and when, not what.`
        : `${workerName(0)} and ${workerName(3)} hold both pieces of your data and both pieces of the model. Together they see exactly what you see. This is where the design stops: it relies on such workers being run by independent operators who do not pool what they hold.`;
  const looking = eye !== "you";
  const showReal = !looking || eyeView.real;

  const overlay = (
    <>
      <div className="show-hud">
        <div className="show-progress" role="progressbar" aria-label="Progress of this run" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><span style={{ width: `${progress}%` }} /></div>
        <dl className="show-counters">
          <div><dt>{words === "plain" ? "Answers checked" : "Products checked"}</dt><dd data-testid="checks">{snapshot.checks}</dd></div>
          <div className={snapshot.failed > 0 ? "bad" : undefined}><dt>Rejected</dt><dd data-testid="rejected">{snapshot.failed}</dd></div>
        </dl>
      </div>
      <div className="show-side" data-inset="right">
        <div className={looking ? (eye === "match" ? "show-panel show-output eye warn" : "show-panel show-output eye") : "show-panel show-output"}>
          <p className="show-label">{eyeEntry.title}</p>
          <p className="show-prompt"><span className="show-tag">{mode === "own-text" || mode === "own-drawing" ? "Yours" : "Prompt"}</span> <q>{model.prompt}</q></p>
          {model.toy && !looking ? <p className="show-toy">{model.toy}</p> : null}
          {model.kind === "text" && showReal ? <p className="show-answer">{snapshot.tokens.join("")}<span className="show-caret" aria-hidden="true">▌</span></p> : null}
          {model.kind === "text" && !showReal ? <pre className="show-eye-data">{eyeView.text || "Nothing yet. Press Play."}</pre> : null}
          <canvas ref={deviceCanvas} className="show-picture" width={96} height={60} hidden={model.kind === "text" || looking} role="img" aria-label={model.kind === "text" ? undefined : `${model.prompt}: ${done ? "finished" : "forming"}`} />
          <canvas ref={eyeCanvas} className="show-picture" width={96} height={60} hidden={model.kind === "text" || !looking} role="img" aria-label={eyeView.real ? "The same picture you see" : "Noise"} />
          {model.kind === "clip" && !looking ? (
            <ol className="show-film" aria-label={`Finished frames: ${snapshot.frames} of ${model.frames.length}`}>
              {model.frames.map((_, index) => (
                <li key={index} className={index < snapshot.frames ? "ready" : undefined}>
                  <canvas ref={(element) => { filmCanvases.current[index] = element; }} width={model.width} height={model.height} aria-hidden="true" />
                </li>
              ))}
            </ol>
          ) : null}
          {looking ? <p className="show-eye-note">{eyeNote}</p> : null}
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
      <div className="show-caption">
        <p>{scene.caption}</p>
        <p className="show-why"><span>Why it matters</span> {showWhy(beat)}</p>
      </div>
    </>
  );

  return (
    <section ref={sectionRef} className={`theater${full ? " full" : ""}${compact ? " compact" : ""}`} aria-labelledby={compact ? undefined : titleId} aria-label={compact ? "Live demonstration with your own text" : undefined}>
      <header className="show-head">
        {compact ? (
          <p className="show-lead"><strong>Try it.</strong> Type a message and watch it go through the six steps. No worker ever sees it.</p>
        ) : (
          <div>
            <span className="evidence">Simulation · toy models · real share arithmetic</span>
            <Heading id={titleId}>Live demonstration</Heading>
            <p>An answer, a picture and a short clip are produced below, one protected step at a time, while the 3D view shows the steps that produce them. Then try it with your own text or drawing.</p>
          </div>
        )}
        <div className="show-pick">
          <div className="playback" role="group" aria-label="What to produce">
            {MODES.map((entry) => (
              <button key={entry.mode} className="show-kind" aria-pressed={entry.mode === mode} onClick={() => selectMode(entry.mode)}>{entry.label}</button>
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
        {mode === "own-text" ? (
          <div className="show-own">
            <label>Type something <input value={ownText} maxLength={40} onChange={(event) => typeOwnText(event.target.value)} /></label>
            <p>It stays in this page. A toy model writes it backwards, one character per step, and each step runs on random pieces.</p>
          </div>
        ) : null}
        {mode === "own-drawing" ? (
          <div className="show-own">
            <canvas
              ref={padCanvas}
              className="show-pad"
              width={PAD.width}
              height={PAD.height}
              role="img"
              aria-label="Drawing pad. Draw with the mouse or a finger, or choose a ready-made drawing."
              onPointerDown={(event) => { drawingRef.current = true; event.currentTarget.setPointerCapture?.(event.pointerId); dab(event); }}
              onPointerMove={(event) => { if (drawingRef.current) dab(event); }}
              onPointerUp={() => { if (drawingRef.current) { drawingRef.current = false; commitInk(inkRef.current); } }}
              onPointerCancel={() => { drawingRef.current = false; }}
            />
            <div>
              <div className="playback" role="group" aria-label="Ready-made drawings">
                {(["smile", "arrow", "blank"] as PadPreset[]).map((preset) => (
                  <button key={preset} className="show-choice" onClick={() => commitInk(presetInk(preset))}>{preset === "blank" ? "Clear" : preset === "smile" ? "Smile" : "Arrow"}</button>
                ))}
              </div>
              <p>Draw on the pad. It stays in this page. A toy model mirrors it and raises its edges, working on random pieces.</p>
            </div>
          </div>
        ) : null}
      </header>

      {compact ? null : (
        <ol className="show-how" aria-label="How to read the scene">
          <li><strong>Your device</strong> (green) keeps your data and cuts it into random pieces.</li>
          <li><strong>Four workers</strong> (blue) each compute on one piece. Their screens show what they see.</li>
          <li><strong>Your device</strong> checks every answer and puts the result together.</li>
        </ol>
      )}

      <div className="show-eyes playback" role="group" aria-label="Whose eyes">
        <span className="show-eyes-title">Whose eyes?</span>
        {EYES.map((entry) => (
          <button key={entry.eye} className="show-choice" aria-pressed={entry.eye === eye} onClick={() => setEye(entry.eye)}>{entry.label}</button>
        ))}
      </div>

      <Scene3DView scene={scene} variant="show" direct={direct} screens={screens} screensVersion={snapshot.version + loopFrame} overlay={overlay} />

      <div className="show-controls playback" role="group" aria-label="Demonstration controls">
        <button className="show-main" onClick={() => dispatch(state.playing ? { type: "pause" } : done ? { type: "restart" } : { type: "play" })}>{state.playing ? "Pause" : "Play"}</button>
        <button onClick={() => dispatch({ type: "next", units: model.units })} disabled={done}>Next step</button>
        <button onClick={() => dispatch({ type: "restart" })}>Restart</button>
        <button onClick={() => dispatch({ type: "speed", speed: state.speed === 1 ? 2 : 1 })}>Speed: {state.speed}×</button>
        <button onClick={() => dispatch({ type: "tamper" })} disabled={done || state.tamperArmed || state.rejecting} title="Change one number in one worker's answer and watch the check catch it.">Tamper with a worker</button>
        <button onClick={() => wordsSetting.set(words === "plain" ? "technical" : "plain")} title="Everyday wording, or the terms the chapters use.">Words: {words}</button>
        <button onClick={toggleFull}>{full ? "Exit full screen" : "Full screen"}</button>
      </div>
      <p className="sr-only" role="status">{announcement}</p>

      <p className="show-truth">
        <strong>Simulation in your browser.</strong> The answer, picture and clip are prepared toy models; the share arithmetic and its checks are real.
        Pace is set for viewing and measures nothing. Designed for defence in depth; claims under evaluation.
      </p>
      {compact ? null : (
        <details className="show-details" open={factsOpen}>
          <summary>What is real here</summary>
          <div className="show-facts">
            <div>
              <FactHeading>Real, computed in this page</FactHeading>
              <ul>
                <li>Every token, picture step and clip pass is one matrix product. Its input is split into two random shares, four roles each multiply one input share by one weight share, and the four products are added.</li>
                <li>Every product is tested with a Freivalds check before anything is combined. “Tamper with a worker” changes one number by one, and that check rejects it.</li>
                <li>The worker tiles and screens show the numbers each role actually holds in the current step, and “Whose eyes?” rebuilds what each party could work out from them.</li>
                <li>What you read and see is decoded from the recombined result, including your own text and drawing.</li>
              </ul>
            </div>
            <div>
              <FactHeading>Prepared</FactHeading>
              <ul>
                <li>No language or image model runs. Text comes from a lookup table that holds one prepared answer per prompt.</li>
                <li>Pictures and the clip come from a small hand-built linear decoder that is driven along a prepared path from noise to the image.</li>
                <li>Your own text is only written backwards and your own drawing only mirrored and embossed: two simple linear maps, chosen because they can be checked by eye.</li>
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
      )}
    </section>
  );
}
