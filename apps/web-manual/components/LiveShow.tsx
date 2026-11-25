"use client";

import Link from "next/link";
import { useEffect, useId, useMemo, useReducer, useRef, useState, useSyncExternalStore } from "react";
import type { FlowForm } from "@/components/FlowRibbon";
import { QuestionPicker } from "@/components/QuestionPicker";
import { ScrambleText } from "@/components/ScrambleText";
import { StageBox } from "@/components/StageBox";
import { wordsSetting } from "@/lib/scene/settings";
import { relayOf, showCamera, showScene, showWhy, workerName } from "@/lib/scene/show-scene";
import { inkToRgb, presetInk, stampInk, type PadPreset } from "@/lib/show/pad";
import { paintBlank, paintCipher, paintNumbers, paintPixels, paintText } from "@/lib/show/paint";
import { useJourney } from "@/lib/show/question";
import { hexRun } from "@/lib/show/scramble";
import { ATTRACT_ORDER, SLOW_BEATS, beatMs, currentBeat, initialShow, showReducer, tamperRoleOf, type Attack, type Beat } from "@/lib/show/timeline";
import { WORKER_CODE, attest, makeEnclave, verifyReport, type Enclave } from "@/lib/simulation/enclave";
import {
  PAD, SHOW_MODELS, SHOW_PRIME, cryptoRng, decodePixels, getModel, openSession, ownDrawingModel, ownTextModel, pooledView, runStep,
  sealForWire, sealRoute, shuffled, workerBytes, type Rng, type ShowSession, type ShowStep, type Shuffled,
} from "@/lib/simulation/show";

type Mode = "text" | "picture" | "clip" | "own-text" | "own-drawing";
const MODES: Array<{ mode: Mode; label: string }> = [
  { mode: "text", label: "Text" },
  { mode: "picture", label: "Picture" },
  { mode: "clip", label: "Clip" },
  { mode: "own-text", label: "Your text" },
  { mode: "own-drawing", label: "Your drawing" },
];

type Eye = "you" | "worker" | "same" | "match" | "relay" | "wire" | "operator";
const EYES: Array<{ eye: Eye; label: string; title: string }> = [
  { eye: "you", label: "You", title: "You get" },
  { eye: "worker", label: "One worker", title: `${workerName(0)} alone` },
  { eye: "same", label: "Two workers, same piece", title: `${workerName(0)} and ${workerName(1)} together` },
  { eye: "match", label: "Two workers, matching pieces", title: `${workerName(0)} and ${workerName(3)} together` },
  { eye: "relay", label: "A Tor relay", title: "Relay 2 of the circuit" },
  { eye: "wire", label: "The network", title: "Someone watching the network" },
  { eye: "operator", label: "The machine's operator", title: "The operator of a worker's machine" },
];

const ATTACKS: Array<{ attack: Attack; label: string; title: string }> = [
  { attack: "result", label: "Tamper with a worker", title: "Change one number in one worker's result and watch the check catch it." },
  { attack: "software", label: "Swap a worker's software", title: "Make one worker run different software and watch its report be refused." },
  { attack: "packet", label: "Alter a sealed packet", title: "Flip one bit of a sealed packet on the way and watch a relay refuse it." },
];

const ROLES = [0, 1, 2, 3];
const RELAYS = [0, 1, 2];
const CLIP_FRAME_MS = 125;

/** Which forms of the data are lit for each step. */
const LIT: Record<Beat, string[]> = {
  split: ["numbers", "shuffled", "pieces"],
  attest: ["enclave"],
  seal: ["sealed"],
  route: ["sealed"],
  compute: ["enclave"],
  return: ["sealed"],
  verify: ["checked"],
  combine: ["answer"],
  stream: ["readable", "numbers", "shuffled", "pieces", "sealed", "enclave", "checked", "answer"],
  rejected: ["checked"],
  done: ["answer"],
};

/** Everything one run has produced so far. Kept outside React state because it holds large buffers. */
interface RunData {
  run: number;
  model: unknown;
  disguised: Shuffled;
  session: ShowSession;
  done: number;
  rejections: number;
  /** How many of those rejections have had their failed check counted. */
  counted: number;
  tokens: string[];
  pixels: Uint8ClampedArray | null;
  /** Running sum of the parts, for models whose output arrives in parts. */
  total: Int32Array | null;
  frames: Uint8ClampedArray[];
  checks: number;
  failed: number;
  last: ShowStep | null;
  /** The first step, worked out when the run opens and not yet revealed, so the slow pass can show its data. */
  ahead: ShowStep | null;
  /** The unit `last` belongs to. */
  lastUnit: number;
}

interface Snapshot {
  tokens: string[];
  frames: number;
  checks: number;
  failed: number;
  /** The current step's data in the forms shown on the ribbon. */
  numbers: string;
  shuffled: string;
  pieces: string;
  result: string;
  passed: string;
  /** Bumped whenever the canvases were repainted. */
  version: number;
}

const EMPTY: Snapshot = { tokens: [], frames: 0, checks: 0, failed: 0, numbers: "", shuffled: "", pieces: "", result: "", passed: "", version: 0 };

/** What another party gets to look at: either raw numbers (as text) or, when it can rebuild it, the real output. */
interface EyeView {
  text: string;
  real: boolean;
  bytes: number;
}

/** What the relays saw of the latest step. */
interface Sealed {
  hex: string;
  bytes: number;
  failedAt: number | null;
  middle: Uint8Array | null;
}

const UNSEALED: Sealed = { hex: "", bytes: 0, failedAt: null, middle: null };

function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * The live demonstration. An answer, picture or clip is produced unit by unit. Every unit is one matrix
 * product whose input is shuffled, split into additive shares, sealed in three layers, sent through three
 * relays and computed by four workers; the results are checked and recombined. All of that arithmetic and
 * cryptography really runs in this page. The relays and enclaves are stand-ins, the models are toys, and
 * nothing here measures speed.
 */
export function LiveShow({ autoPlay = false, heading = "h2", factsOpen = false, compact = false, rng }: {
  autoPlay?: boolean;
  heading?: "h1" | "h2";
  factsOpen?: boolean;
  /** A smaller version for use inside a chapter: it follows the question chosen for the labs and diagrams. */
  compact?: boolean;
  rng?: Rng;
}) {
  const titleId = useId();
  const [state, dispatch] = useReducer(showReducer, undefined, () => initialShow(ATTRACT_ORDER[0]!, { playing: false, attract: autoPlay && !compact }));
  const chosen = useJourney().id;
  const [ownText, setOwnText] = useState("hello world");
  const [ownInk, setOwnInk] = useState(() => presetInk("smile"));
  const model = useMemo(
    () => (state.modelId === "own-text" ? ownTextModel(ownText) : state.modelId === "own-drawing" ? ownDrawingModel(ownInk) : getModel(state.modelId)),
    [state.modelId, ownText, ownInk],
  );
  const mode: Mode = model.id === "own-text" || model.id === "own-drawing" ? model.id : model.kind;
  const beat = currentBeat(state, model.units);
  const tamperRole = tamperRoleOf(state);
  const rejectKind = state.rejecting ? state.rejectKind : null;
  const words = useSyncExternalStore(wordsSetting.subscribe, wordsSetting.get, () => "plain" as const);
  // During the slow first pass the first step's data is shown form by form, each when its step is reached.
  const firstPass = state.revealed === 0 && state.slow && beat !== "done";
  const reached = SLOW_BEATS.indexOf(beat);
  const past = (step: Beat) => !firstPass || reached < 0 || reached >= SLOW_BEATS.indexOf(step);
  const showSealed = past("seal");
  const showRouted = past("route");
  const showWork = past("compute");
  const showChecked = past("verify");

  const sectionRef = useRef<HTMLElement>(null);
  const deviceCanvas = useRef<HTMLCanvasElement>(null);
  const eyeCanvas = useRef<HTMLCanvasElement>(null);
  const padCanvas = useRef<HTMLCanvasElement>(null);
  const workerCanvases = useRef<Array<HTMLCanvasElement | null>>([null, null, null, null]);
  const relayCanvases = useRef<HTMLCanvasElement[]>([]);
  const filmCanvases = useRef<Array<HTMLCanvasElement | null>>([]);
  const dataRef = useRef<RunData | null>(null);
  const rngRef = useRef<Rng | null>(rng ?? null);
  const inkRef = useRef<Uint8ClampedArray>(ownInk);
  const drawingRef = useRef(false);
  const enclavesRef = useRef<Promise<Enclave[]> | null>(null);

  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY);
  const [screens, setScreens] = useState<Record<string, HTMLCanvasElement | null>>({});
  const [onScreen, setOnScreen] = useState(true);
  const [tabVisible, setTabVisible] = useState(true);
  const [full, setFull] = useState(false);
  const [loopFrame, setLoopFrame] = useState(0);
  const [eye, setEye] = useState<Eye>("you");
  const [eyeView, setEyeView] = useState<EyeView>({ text: "", real: false, bytes: 0 });
  const [sealed, setSealed] = useState<Sealed>(UNSEALED);
  const [sealedVersion, setSealedVersion] = useState(0);
  const [attested, setAttested] = useState<boolean[] | null>(null);
  const [workerPaint, setWorkerPaint] = useState(0);
  const routeRef = useRef<{ step: ShowStep; alteredAt: number | null; route: ReturnType<typeof sealRoute> } | null>(null);

  // The canvases double as the pictures on the 3D view's screens. The relays' screens have no place on the page.
  useEffect(() => {
    relayCanvases.current = RELAYS.map(() => document.createElement("canvas"));
    relayCanvases.current.forEach((canvas) => paintCipher(canvas, null));
    setScreens({
      device: deviceCanvas.current,
      "worker-0": workerCanvases.current[0]!, "worker-1": workerCanvases.current[1]!, "worker-2": workerCanvases.current[2]!, "worker-3": workerCanvases.current[3]!,
      "relay-0": relayCanvases.current[0]!, "relay-1": relayCanvases.current[1]!, "relay-2": relayCanvases.current[2]!,
    });
  }, []);

  // Inside a chapter the demonstration follows the question chosen for the labs and diagrams.
  useEffect(() => {
    if (compact) dispatch({ type: "select", modelId: chosen, playing: !prefersReducedMotion() });
  }, [compact, chosen]);

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
    if (!data || data.run !== state.run || data.model !== model) {
      // The numbers are shuffled with a fresh secret order for every run, and the model's table with them.
      const disguised = shuffled(model, rngRef.current);
      data = { run: state.run, model, disguised, session: openSession(disguised, rngRef.current), done: 0, rejections: 0, counted: 0, tokens: [], pixels: null, total: null, frames: [], checks: 0, failed: 0, last: null, ahead: null, lastUnit: 0 };
      dataRef.current = data;
      if (device) {
        if (model.kind === "text") paintText(device, model.prompt, "");
        else paintBlank(device, model.width, model.height);
      }
      for (const canvas of filmCanvases.current) if (canvas && model.kind !== "text") paintBlank(canvas, model.width, model.height);
    }
    // The first step is worked out as soon as the run opens. It is counted, and its result appears, only
    // when the timeline reveals it.
    if (data.done === 0 && !data.ahead) data.ahead = runStep(data.session, 0);
    let last: ShowStep | null = null;
    let unit = data.lastUnit;
    while (data.done < state.revealed && data.done < model.units) {
      unit = data.done;
      last = unit === 0 && data.ahead ? data.ahead : runStep(data.session, unit);
      data.ahead = null;
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
      unit = data.done;
      if (rejectKind === "result") {
        // One worker's result is changed by one, and the real check must catch it.
        last = runStep(data.session, unit, tamperRole);
        data.checks += last.roles.length;
        data.failed += last.roles.filter((role) => !role.valid).length;
        data.counted = state.rejections;
      } else {
        // Swapped software or an altered packet: the step is computed honestly here; the refusal is
        // established by the signature and authenticity checks below, which count it when they fail.
        last = runStep(data.session, unit);
      }
      data.rejections = state.rejections;
    }
    if (last) {
      data.last = last;
      data.lastUnit = unit;
      if (device && model.kind === "text") paintText(device, model.prompt, data.tokens.join(""));
      if (device && model.kind !== "text" && data.pixels) paintPixels(device, data.pixels, model.width, model.height);
    }
    const produced = data;
    const shown = produced.last ?? produced.ahead;
    setSnapshot((current) => ({
      tokens: [...produced.tokens],
      frames: produced.frames.length,
      checks: produced.checks,
      failed: produced.failed,
      numbers: shown ? hexRun(model.input(produced.lastUnit), 4, 6) : "",
      shuffled: shown ? hexRun(produced.disguised.input(produced.lastUnit), 4, 6) : "",
      pieces: shown ? `${hexRun(shown.activationShares[0], 4, 3)} | ${hexRun(shown.activationShares[1], 4, 3)}` : "",
      result: shown ? hexRun(shown.roles[0]!.product, 4, 5) : "",
      passed: shown ? `${shown.roles.filter((role) => role.valid).length} of 4 pass` : "",
      version: current.version + 1,
    }));
  }, [state.run, state.revealed, state.rejections, tamperRole, rejectKind, model]);

  // The workers' screens show the numbers each one holds, from the moment the workers have them.
  useEffect(() => {
    const data = dataRef.current;
    const step = showWork ? data?.last ?? data?.ahead : null;
    workerCanvases.current.forEach((canvas, index) => {
      if (!canvas) return;
      const role = step?.roles[index];
      if (!role) paintBlank(canvas, 64, 40);
      else if (model.kind === "text") paintNumbers(canvas, role.product, !role.valid);
      else paintPixels(canvas, workerBytes(role.product), model.width, model.height);
    });
    setWorkerPaint((current) => current + 1);
  }, [snapshot.version, showWork, model]);

  // Sealing: the first piece of the latest step really is wrapped in three layers and sent through three relays.
  const alteredAt = rejectKind === "packet" && tamperRole !== null ? relayOf(tamperRole) : null;
  useEffect(() => {
    const data = dataRef.current;
    const last = data?.last ?? data?.ahead;
    if (!data || !last) {
      relayCanvases.current.forEach((canvas) => paintCipher(canvas, null));
      setSealed(UNSEALED);
      setSealedVersion((current) => current + 1);
      return;
    }
    let current = true;
    const counted = state.rejections;
    // One sealing per step: showing it at a later moment must not seal it again under new keys.
    const kept = routeRef.current;
    const sealing = kept && kept.step === last && kept.alteredAt === alteredAt ? kept.route : sealRoute(last.activationShares[0], alteredAt ?? undefined);
    routeRef.current = { step: last, alteredAt, route: sealing };
    void sealing.then((route) => {
      if (!current) return;
      // The relays have nothing until the sealed piece is sent.
      relayCanvases.current.forEach((canvas, index) => paintCipher(canvas, showRouted ? route.views[index] ?? null : null));
      setSealed({ hex: hexRun(route.views[0]!, 2, 10), bytes: route.views[0]!.length, failedAt: route.failedAt, middle: route.views[1] ?? route.views[0]! });
      setSealedVersion((version) => version + 1);
      if (route.failedAt !== null && dataRef.current === data && data.counted < counted) {
        data.failed += 1;
        data.counted = counted;
        setSnapshot((snap) => ({ ...snap, failed: data.failed }));
      }
    });
    return () => {
      current = false;
    };
  }, [snapshot.version, alteredAt, state.rejections, showRouted]);

  // Attestation: every enclave signs a report over its code and a fresh nonce, and the device verifies it.
  const swappedRole = rejectKind === "software" ? tamperRole : null;
  useEffect(() => {
    if (typeof crypto === "undefined" || !crypto.subtle) return;
    let current = true;
    const data = dataRef.current;
    const counted = state.rejections;
    void (async () => {
      enclavesRef.current ??= Promise.all(ROLES.map(() => makeEnclave()));
      const enclaves = await enclavesRef.current;
      const nonce = crypto.getRandomValues(new Uint8Array(16));
      const checks = await Promise.all(enclaves.map(async (enclave, role) => verifyReport(await attest(enclave, nonce, role === swappedRole ? `${WORKER_CODE} + something else` : undefined), enclave.publicKey, nonce)));
      if (!current) return;
      setAttested(checks.map((check) => check.ok));
      if (swappedRole !== null && checks.some((check) => !check.ok) && data && dataRef.current === data && data.counted < counted) {
        data.failed += 1;
        data.counted = counted;
        setSnapshot((snap) => ({ ...snap, failed: data.failed }));
      }
    })();
    return () => {
      current = false;
    };
  }, [state.run, swappedRole, state.rejections]);

  // Through someone else's eyes: rebuilt from what that party really holds in the latest step.
  useEffect(() => {
    if (eye === "you") return;
    const data = dataRef.current;
    const canvas = eyeCanvas.current;
    const last = data?.last ?? (showWork ? data?.ahead : null);
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
      setEyeView({ text: hexRun(values, digits, 28), real: false, bytes });
    };
    if (eye === "worker") {
      showNumbers(last.roles[0]!.product, 4, 0);
    } else if (eye === "same") {
      showNumbers(pooledView(data.session, last, 0, 1).sum, 4, 0);
    } else if (eye === "relay") {
      if (sealed.middle) showNumbers(sealed.middle, 2, sealed.middle.length);
    } else if (eye === "wire" || eye === "operator") {
      void sealForWire(last.roles[0]!.product).then((bytes) => showNumbers(bytes, 2, bytes.length));
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
  }, [eye, snapshot.version, sealed, model, showWork]);

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
    () => showScene({ kind: model.kind, beat, revealed: state.revealed, units: model.units, rejections: state.rejections, tamperRole, rejectKind, words }),
    [model, beat, state.revealed, state.rejections, tamperRole, rejectKind, words],
  );
  const direct = useMemo(() => showCamera(beat, tamperRole, rejectKind), [beat, tamperRole, rejectKind]);

  const Heading = heading;
  const FactHeading = heading === "h1" ? "h2" : "h3";
  const done = beat === "done";
  const choices = SHOW_MODELS.filter((entry) => entry.kind === model.kind && mode === model.kind);
  const progress = Math.round((100 * Math.min(state.revealed, model.units)) / model.units);
  const refusal = tamperRole === null ? ""
    : rejectKind === "software" ? `${workerName(tamperRole)}'s report was refused. Nothing was sent to it; a checked worker takes over.`
      : rejectKind === "packet" ? `Relay ${relayOf(tamperRole) + 1} refused an altered packet. It was dropped and sent again.`
        : `${workerName(tamperRole)} failed the check. Step rejected; sending it again.`;
  const announcement = tamperRole !== null ? refusal
    : !done ? ""
      : model.kind === "text" ? `Answer complete: ${model.answer}`
        : model.kind === "picture" ? "Picture complete." : "Clip complete.";

  const eyeEntry = EYES.find((entry) => entry.eye === eye)!;
  const eyeNote = eye === "worker" ? `${workerName(0)} holds one piece of your data and one piece of the model. This is everything it has.`
    : eye === "same" ? `${workerName(0)} and ${workerName(1)} hold the same piece of your data. Together they still have only noise.`
      : eye === "relay" ? `The middle relay sees ${eyeView.bytes} sealed bytes and the next stop. It cannot read them, and it does not know who sent them or which worker will get them.`
        : eye === "wire" ? `Someone watching the network sees ${eyeView.bytes} encrypted bytes (AES-GCM here). They learn how much was sent and when, not what.`
          : eye === "operator" ? "The company that runs the machine sees an enclave it cannot look into; the sealed bytes here stand in for that. This rests on the processor and its maker, which is why an enclave counts as one more layer and not as proof."
            : `${workerName(0)} and ${workerName(3)} hold both pieces of your data and both pieces of the model. Together they see exactly what you see. This is where the design stops: it relies on such workers being run by independent operators who do not pool what they hold.`;
  const looking = eye !== "you";
  const showReal = !looking || eyeView.real;

  const forms: FlowForm[] = [
    { id: "readable", label: "Readable", value: model.prompt.length > 26 ? `${model.prompt.slice(0, 25)}…` : model.prompt, tone: "good" },
    { id: "numbers", label: "Numbers", value: snapshot.numbers || "—", tone: "plain" },
    { id: "shuffled", label: "Shuffled", value: snapshot.shuffled || "—", tone: "plain" },
    { id: "pieces", label: "Two pieces", value: snapshot.pieces || "—", tone: "accent" },
    { id: "sealed", label: sealed.failedAt === null ? "Sealed ×3" : "Refused", value: (showSealed && sealed.hex) || "—", tone: sealed.failedAt === null ? "cipher" : "warn" },
    { id: "enclave", label: "In enclave", value: (showWork && snapshot.result) || "—", tone: attested && attested.some((ok) => !ok) ? "warn" : "accent" },
    { id: "checked", label: "Checked", value: rejectKind === "result" ? "1 of 4 fails" : (showChecked && snapshot.passed) || "—", tone: rejectKind === "result" ? "warn" : "good" },
    { id: "answer", label: "Answer", value: model.kind === "text" ? (snapshot.tokens.join("").slice(-22) || "—") : `${Math.min(state.revealed, model.units)} of ${model.units} steps`, tone: "good" },
  ];
  const lit = beat === "rejected" ? [rejectKind === "software" ? "enclave" : rejectKind === "packet" ? "sealed" : "checked"] : LIT[beat];

  const input = (
    <div className="stage-panel stage-in">
      <p className="stage-label">{mode === "own-text" || mode === "own-drawing" ? "You put in" : "You ask"}</p>
      <p className="show-prompt"><span className="show-tag">{mode === "own-text" || mode === "own-drawing" ? "Yours" : "Prompt"}</span> <q><ScrambleText text={model.prompt} /></q></p>
      {model.toy ? <p className="show-toy">{model.toy}</p> : null}
      <p className="stage-leaves"><span>What leaves your device</span> <ScrambleText className="stage-cipher" text={(showRouted && sealed.hex) || "nothing yet"} /></p>
    </div>
  );

  const output = (
    <div className={looking ? (eye === "match" ? "stage-panel stage-out show-output eye warn" : "stage-panel stage-out show-output eye") : "stage-panel stage-out show-output"}>
      <p className="stage-label">{eyeEntry.title}</p>
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
      {tamperRole !== null ? <p className="show-rejected">{refusal}</p> : null}
    </div>
  );

  const corner = (
    <>
    <div className="stage-panel show-hud">
      <div className="show-progress" role="progressbar" aria-label="Progress of this run" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><span style={{ width: `${progress}%` }} /></div>
      <dl className="show-counters">
        <div><dt>{words === "plain" ? "Answers checked" : "Products checked"}</dt><dd data-testid="checks">{snapshot.checks + (firstPass && reached >= 0 && showChecked ? ROLES.length : 0)}</dd></div>
        <div className={snapshot.failed > 0 ? "bad" : undefined}><dt>Rejected</dt><dd data-testid="rejected">{snapshot.failed}</dd></div>
      </dl>
    </div>
    <div className="stage-panel">
      <div className="show-workers">
        <p className="stage-label">What each worker sees</p>
        <ul>
          {ROLES.map((role) => (
            <li key={role} className={tamperRole === role && rejectKind !== "packet" ? "show-tile bad" : "show-tile"}>
              <canvas ref={(element) => { workerCanvases.current[role] = element; }} width={64} height={40} aria-hidden="true" />
              <span><span className="sr-only">Worker </span>({role >> 1},{role & 1}){attested ? (attested[role] ? " ✓" : " ✗") : ""}</span>
            </li>
          ))}
        </ul>
        <p className="show-note">Real numbers from this step: noise. A tick means the worker&apos;s signed report was verified.</p>
      </div>
    </div>
    </>
  );

  return (
    <section ref={sectionRef} className={`theater${full ? " full" : ""}${compact ? " compact" : ""}`} aria-labelledby={compact ? undefined : titleId} aria-label={compact ? "Live demonstration of the chosen question" : undefined}>
      <header className="show-head">
        {compact ? (
          <div className="show-lead">
            <p><strong>Watch it live.</strong> Choose a question and follow it through the eight steps to the model&apos;s answer. No worker ever sees it.</p>
            <QuestionPicker />
          </div>
        ) : (
          <div>
            <span className="evidence">Simulation · toy models · real arithmetic and encryption</span>
            <Heading id={titleId}>Live demonstration</Heading>
            <p>Watch a question get shuffled, split into random pieces, sealed in three layers, routed through relays, computed inside enclaves, checked and put back together. It keeps running; then try your own text or drawing.</p>
          </div>
        )}
        {compact ? null : <div className="show-pick">
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
        </div>}
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

      <div className="show-eyes playback" role="group" aria-label="Whose eyes">
        <span className="show-eyes-title">Whose eyes?</span>
        {EYES.map((entry) => (
          <button key={entry.eye} className="show-choice" aria-pressed={entry.eye === eye} onClick={() => setEye(entry.eye)}>{entry.label}</button>
        ))}
      </div>

      <StageBox
        label="The pipeline, running"
        scene={scene}
        direct={direct}
        screens={screens}
        screensVersion={snapshot.version + loopFrame + sealedVersion + workerPaint}
        left={<>{input}{output}</>}
        right={corner}
        caption={scene.caption}
        why={showWhy(beat, rejectKind)}
        forms={forms}
        current={lit}
      />

      <div className="show-controls playback" role="group" aria-label="Demonstration controls">
        <button className="show-main" onClick={() => dispatch(state.playing ? { type: "pause" } : done ? { type: "restart" } : { type: "play" })}>{state.playing ? "Pause" : "Play"}</button>
        <button onClick={() => dispatch({ type: "next", units: model.units })} disabled={done}>Next step</button>
        <button onClick={() => dispatch({ type: "restart" })}>Restart</button>
        <button onClick={() => dispatch({ type: "speed", speed: state.speed === 1 ? 2 : 1 })}>Speed: {state.speed}×</button>
        <button onClick={() => wordsSetting.set(words === "plain" ? "technical" : "plain")} title="Everyday wording, or the terms the chapters use.">Words: {words}</button>
        <button onClick={toggleFull}>{full ? "Exit full screen" : "Full screen"}</button>
      </div>
      <div className="show-attacks playback" role="group" aria-label="Try to break it">
        <span className="show-eyes-title">Try to break it:</span>
        {ATTACKS.map((entry) => (
          <button key={entry.attack} title={entry.title} onClick={() => dispatch({ type: "tamper", attack: entry.attack })} disabled={done || state.armed !== null || state.rejecting}>{entry.label}</button>
        ))}
      </div>
      <p className="sr-only" role="status">{announcement}</p>

      <p className="show-truth">
        <strong>Simulation in your browser.</strong> The answer, picture and clip are prepared toy models, and the relays and enclaves are stand-ins inside this page;
        the share arithmetic, the three layers of encryption and the signature checks are real. Pace is set for viewing and measures nothing.
        Designed for defence in depth; claims under evaluation.
      </p>
      {compact ? null : (
        <details className="show-details" open={factsOpen}>
          <summary>What is real here</summary>
          <div className="show-facts">
            <div>
              <FactHeading>Real, computed in this page</FactHeading>
              <ul>
                <li>Every token, picture step and clip pass is one matrix product. Its input is shuffled with a secret order, split into two random shares, four roles each multiply one input share by one weight share, and the four products are added.</li>
                <li>Every product is tested with a Freivalds check before anything is combined. “Tamper with a worker” changes one number by one, and that check rejects it.</li>
                <li>A piece of every step is wrapped in three layers of AES-GCM and passed through three relays, each removing one layer. The relays&apos; screens show the bytes they receive, and “Alter a sealed packet” flips one bit, which the relay&apos;s authenticity check catches.</li>
                <li>Each worker signs a report over the hash of its code and a fresh nonce (ECDSA), and the device verifies it. “Swap a worker&apos;s software” changes that code, and the verification fails.</li>
                <li>“Whose eyes?” rebuilds what each party could work out from what it holds. What you read and see is decoded from the recombined result, including your own text and drawing.</li>
              </ul>
            </div>
            <div>
              <FactHeading>Prepared or simulated</FactHeading>
              <ul>
                <li>No language or image model runs. Text comes from a lookup table that holds one prepared answer per prompt; pictures and the clip come from a small hand-built linear decoder driven along a prepared path from noise to the image.</li>
                <li>The three relays stand in for a Tor circuit. They are inside this page, so nothing here shows that the real network hides who you are.</li>
                <li>The enclaves are simulated. A real enclave&apos;s key is held by the processor and vouched for by its maker; here it lives in the page. Hardware attestation of workers has not been validated yet, and an enclave is one more layer, not proof.</li>
                <li>The four workers are four roles inside this page, not four machines. Arithmetic uses a 16-bit prime so that it stays exact at picture size; the engine works in a larger ring.</li>
                <li>Shuffling is disguise, not encryption. The pace is chosen for viewing and measures nothing.</li>
              </ul>
            </div>
            <div>
              <FactHeading>Not shown</FactHeading>
              <ul>
                <li>How the real relay layers are keyed: see the <Link href="/laboratory#onion">onion laboratory</Link> and <Link href="/08-secure-transport">secure transport</Link>.</li>
                <li>Recovery when a worker fails: see the <Link href="/laboratory#pipeline">pipeline laboratory</Link>.</li>
                <li>Wiping memory after use: see the <Link href="/laboratory#memory">memory laboratory</Link>.</li>
                <li>What happens when workers pool their shares, and the limits of enclaves: see the <Link href="/03-threat-model">threat model</Link>.</li>
              </ul>
            </div>
          </div>
        </details>
      )}
    </section>
  );
}
