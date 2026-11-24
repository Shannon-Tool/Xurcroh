"use client";

import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { declutter, type LabelPoint } from "@/lib/scene/labels";
import { DEFAULT_ORBIT, followDirection, orbitMoving, restingMotion, rotateOrbit, stepOrbit, zoomOrbit, type Direction, type Orbit, type OrbitMotion } from "@/lib/scene/orbit";
import { blendFactor } from "@/lib/scene/motion";
import { isSlow, pickQuality, type EffectsPreference } from "@/lib/scene/quality";
import { effectsSetting, motionSetting } from "@/lib/scene/settings";
import type { Palette, Stage } from "@/lib/scene/stage";
import { EMPTY_LIVE, advanceScene, type LiveScene } from "@/lib/scene/tween";
import type { Scene3D, SceneTone } from "@/lib/scene/types";

type Status = "waiting" | "loading" | "ready" | "unsupported" | "lost";

const FALLBACK = { plain: "#8a99a8", accent: "#075985", good: "#15803d", warn: "#9a3412", muted: "#c3ccd6", text: "#17212b", grid: "#ccd5de", top: "#edf1f5", bottom: "#f6f7f9" };
const TONE_TOKENS: Record<SceneTone, string> = { plain: "--muted", accent: "--accent", good: "--good", warn: "--warning", muted: "--border" };
const NEXT_EFFECTS: Record<EffectsPreference, EffectsPreference> = { auto: "on", on: "off", off: "auto" };

/** Colours come from the frame's own CSS tokens, so a view inside a dark band is dark in either theme. */
function readPalette(frame: Element | null): Palette {
  const style = getComputedStyle(frame ?? document.documentElement);
  const token = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    plain: token("--muted", FALLBACK.plain),
    accent: token("--accent", FALLBACK.accent),
    good: token("--good", FALLBACK.good),
    warn: token("--warning", FALLBACK.warn),
    muted: token("--border", FALLBACK.muted),
    text: token("--text", FALLBACK.text),
    grid: token("--border", FALLBACK.grid),
    backdropTop: token("--surface-alt", FALLBACK.top),
    backdropBottom: token("--bg", FALLBACK.bottom),
    dark: document.documentElement.dataset.theme === "dark" || frame?.closest(".theater") != null,
  };
}

function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Interactive 3D drawing of a lab's current state. It animates between states, but the lab's own
 * text and tables remain the exact view, and everything here works without WebGL or motion.
 */
export function Scene3DView({ scene, autoRotate = true, variant = "lab", direct, screens, screensVersion = 0, overlay }: {
  scene: Scene3D;
  autoRotate?: boolean;
  variant?: "lab" | "hero" | "show";
  /** A shot the view settles on until the reader steers it; "Reset view" hands control back. */
  direct?: Direction;
  /** Pictures for the scene's screens, by name. Bump `screensVersion` after redrawing them. */
  screens?: Record<string, HTMLCanvasElement | null>;
  screensVersion?: number;
  /** Panels laid over the view (or under it on small screens). */
  overlay?: React.ReactNode;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const labelLayerRef = useRef<HTMLDivElement>(null);
  const labelElements = useRef(new Map<string, HTMLSpanElement>());
  const stageRef = useRef<Stage | null>(null);
  const liveRef = useRef<LiveScene>(EMPTY_LIVE);
  const motionRef = useRef<OrbitMotion>(restingMotion(DEFAULT_ORBIT));
  const sizeRef = useRef({ width: 640, height: 460 });
  const timeRef = useRef(0);
  const slowRef = useRef(false);
  const touchedRef = useRef(false);
  const nearRef = useRef(typeof IntersectionObserver === "undefined");
  const focusRef = useRef<string | null>(null);
  const hoverRef = useRef<string | null>(null);
  const requestRef = useRef<() => void>(() => undefined);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const dragRef = useRef<{ startX: number; startY: number; lastTime: number; moved: boolean } | null>(null);
  const pinchRef = useRef(0);

  const [status, setStatus] = useState<Status>("waiting");
  // Without IntersectionObserver every view counts as near; with it, wait to be told.
  const [near, setNear] = useState(() => typeof IntersectionObserver === "undefined");
  const [generation, setGeneration] = useState(0);
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  const effects = useSyncExternalStore(effectsSetting.subscribe, effectsSetting.get, () => "auto" as const);
  const motion = useSyncExternalStore(motionSetting.subscribe, motionSetting.get, () => "on" as const);
  const effectsRef = useRef(effects);
  effectsRef.current = effects;
  const directRef = useRef(direct);
  directRef.current = direct;
  const directKey = direct ? `${direct.theta}|${direct.phi}|${direct.zoom}|${direct.focusId}|${direct.sway}` : "";
  const motionOnRef = useRef(motion === "on");
  motionOnRef.current = motion === "on";

  // Labs rebuild their scene object on every render; react only when its content changes.
  const sceneKey = useMemo(() => JSON.stringify(scene), [scene]);
  const sceneRef = useRef(scene);
  sceneRef.current = scene;

  // Only labs near the viewport hold a renderer: browsers allow few WebGL contexts per page.
  useEffect(() => {
    const frame = frameRef.current;
    if (!frame || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.some((entry) => entry.isIntersecting);
      nearRef.current = visible;
      if (visible) setGeneration((current) => current + 1);
      setNear(visible);
    }, { rootMargin: "400px 0px" });
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!near) {
      setStatus("waiting");
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (typeof WebGLRenderingContext === "undefined") {
      setStatus("unsupported");
      return;
    }
    let cancelled = false;
    let stage: Stage | null = null;
    const onLost = (event: Event) => {
      event.preventDefault();
      setStatus("lost");
    };
    canvas.addEventListener("webglcontextlost", onLost);
    setStatus("loading");
    void import("@/lib/scene/stage").then(({ createStage }) => {
      if (cancelled) return;
      stage = createStage(canvas);
      stageRef.current = stage;
      liveRef.current = EMPTY_LIVE;
      setStatus("ready");
    }).catch(() => {
      if (!cancelled) setStatus("unsupported");
    });
    return () => {
      cancelled = true;
      canvas.removeEventListener("webglcontextlost", onLost);
      stage?.dispose();
      stageRef.current = null;
    };
  }, [near, generation]);

  // The drawing loop. It runs only while something is moving and the view is on screen.
  useEffect(() => {
    if (status !== "ready") return;
    const labels = labelElements.current;
    const samples: number[] = [];
    let handle = 0;
    let last = 0;

    const paintLabels = (points: LabelPoint[]) => {
      const layer = labelLayerRef.current;
      if (!layer) return;
      const keep = new Set<string>();
      for (const point of declutter(points, hoverRef.current ?? focusRef.current)) {
        keep.add(point.id);
        let element = labels.get(point.id);
        if (!element) {
          element = document.createElement("span");
          labels.set(point.id, element);
          layer.append(element);
        }
        if (element.textContent !== point.text) element.textContent = point.text;
        element.className = point.active ? "scene3d-label active" : "scene3d-label";
        element.style.transform = `translate(${point.x.toFixed(1)}px, ${point.y.toFixed(1)}px) translate(-50%, -100%)`;
      }
      for (const [id, element] of labels) {
        if (keep.has(id)) continue;
        element.remove();
        labels.delete(id);
      }
    };

    const frame = (now: number) => {
      handle = 0;
      const stage = stageRef.current;
      if (!stage) return;
      const still = !motionOnRef.current || prefersReducedMotion();
      const elapsed = last === 0 ? 0 : Math.min((now - last) / 1000, 0.1);
      last = now;
      if (!still) timeRef.current += elapsed;
      liveRef.current = advanceScene(liveRef.current, sceneRef.current, still ? Infinity : elapsed);
      // The director only moves the camera while motion is on; otherwise the view stays where it is.
      const directed = touchedRef.current || still ? undefined : directRef.current;
      const turning = !still && !touchedRef.current && (directed ? directed.sway > 0 : autoRotate);
      let settling = false;
      if (directed) {
        const before = motionRef.current.orbit;
        const orbit = followDirection(before, directed, timeRef.current, blendFactor(elapsed, 1.8));
        settling = Math.abs(orbit.theta - before.theta) + Math.abs(orbit.phi - before.phi) + Math.abs(orbit.zoom - before.zoom) > 0.0004;
        motionRef.current = restingMotion(orbit);
      } else {
        motionRef.current = still ? restingMotion(motionRef.current.orbit) : stepOrbit(motionRef.current, elapsed, turning);
      }
      const result = stage.draw(liveRef.current, {
        orbit: motionRef.current.orbit,
        width: sizeRef.current.width,
        height: sizeRef.current.height,
        time: still ? 0 : timeRef.current,
        dt: still ? Infinity : elapsed,
        focusId: directed ? directed.focusId : focusRef.current,
      });
      paintLabels(result.labels);
      if (!still && elapsed > 0 && !slowRef.current) {
        samples.push(elapsed * 1000);
        if (samples.length > 60) samples.shift();
        if (isSlow(samples)) {
          slowRef.current = true;
          stage.setQuality(pickQuality({ preference: effectsRef.current, rendererName: stage.rendererName, slow: true }));
        }
      }
      const more = !still && nearRef.current && !document.hidden && (result.animating || !liveRef.current.settled || turning || settling || orbitMoving(motionRef.current));
      if (more) handle = window.requestAnimationFrame(frame);
      else last = 0;
    };

    requestRef.current = () => {
      if (handle === 0) handle = window.requestAnimationFrame(frame);
    };
    const onVisibility = () => requestRef.current();
    document.addEventListener("visibilitychange", onVisibility);
    requestRef.current();
    return () => {
      window.cancelAnimationFrame(handle);
      document.removeEventListener("visibilitychange", onVisibility);
      requestRef.current = () => undefined;
      labels.forEach((element) => element.remove());
      labels.clear();
    };
  }, [status, autoRotate]);

  // Theme, quality and scene changes: hand them to the stage and ask for a frame.
  useEffect(() => {
    if (status !== "ready" || !stageRef.current) return;
    const stage = stageRef.current;
    const apply = () => {
      stage.setPalette(readPalette(frameRef.current));
      requestRef.current();
    };
    apply();
    const theme = new MutationObserver(apply);
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => theme.disconnect();
  }, [status]);

  useEffect(() => {
    if (status !== "ready" || !stageRef.current) return;
    stageRef.current.setQuality(pickQuality({ preference: effects, rendererName: stageRef.current.rendererName, slow: slowRef.current }));
    requestRef.current();
  }, [status, effects]);

  useEffect(() => {
    requestRef.current();
  }, [sceneKey, motion, directKey]);

  // Pictures for the scene's screens: hand them over, and again whenever they were redrawn.
  useEffect(() => {
    if (status !== "ready" || !stageRef.current || !screens) return;
    for (const [id, source] of Object.entries(screens)) if (source) stageRef.current.setScreen(id, source);
    requestRef.current();
  }, [status, screens, screensVersion]);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;
    const measure = () => {
      // In the show the frame also holds panels, so the drawing area is the canvas itself.
      const box = ((variant === "show" ? frame.querySelector("canvas:not([hidden])") : null) ?? frame).getBoundingClientRect();
      if (box.width > 0 && box.height > 0) sizeRef.current = { width: Math.round(box.width), height: Math.round(box.height) };
      requestRef.current();
    };
    measure();
    const resize = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    resize?.observe(frame);
    // Zooming needs Ctrl (or a trackpad pinch) so that an ordinary scroll still moves the page.
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      turn((orbit) => zoomOrbit(orbit, Math.exp(event.deltaY * 0.0015)));
    };
    frame.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      resize?.disconnect();
      frame.removeEventListener("wheel", onWheel);
    };
  }, [variant, status]);

  function turn(change: (orbit: Orbit) => Orbit, spin?: { theta: number; phi: number }) {
    touchedRef.current = true;
    motionRef.current = { orbit: change(motionRef.current.orbit), spinTheta: spin?.theta ?? 0, spinPhi: spin?.phi ?? 0 };
    requestRef.current();
  }

  function resetView() {
    focusRef.current = null;
    if (directRef.current) {
      // Hand the camera back to the scene's director.
      touchedRef.current = false;
      motionRef.current = restingMotion(motionRef.current.orbit);
      requestRef.current();
      return;
    }
    turn(() => DEFAULT_ORBIT);
  }

  function localPoint(event: React.PointerEvent): { x: number; y: number } {
    const box = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - box.left, y: event.clientY - box.top };
  }

  function onPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    dragRef.current = { startX: event.clientX, startY: event.clientY, lastTime: event.timeStamp, moved: false };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    turn((orbit) => orbit);
  }

  function onPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    const previous = pointers.current.get(event.pointerId);
    if (!previous || !dragRef.current) {
      if (event.pointerType !== "mouse" || !stageRef.current) return;
      const point = localPoint(event);
      const id = stageRef.current.pick(point.x, point.y);
      hoverRef.current = id;
      setHover(id ? { id, x: point.x, y: point.y } : null);
      requestRef.current();
      return;
    }
    const deltaX = event.clientX - previous.x;
    const deltaY = event.clientY - previous.y;
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const drag = dragRef.current;
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 5) drag.moved = true;
    if (pointers.current.size >= 2) {
      const [first, second] = [...pointers.current.values()];
      const spread = Math.hypot(first!.x - second!.x, first!.y - second!.y);
      if (pinchRef.current > 0 && spread > 0) turn((orbit) => zoomOrbit(orbit, pinchRef.current / spread));
      pinchRef.current = spread;
      return;
    }
    const seconds = Math.max((event.timeStamp - drag.lastTime) / 1000, 1 / 240);
    drag.lastTime = event.timeStamp;
    const clamp = (value: number) => Math.max(-6, Math.min(6, value));
    turn((orbit) => rotateOrbit(orbit, -deltaX * 0.01, -deltaY * 0.01), { theta: clamp((-deltaX * 0.01) / seconds), phi: clamp((-deltaY * 0.01) / seconds) });
  }

  function onPointerUp(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    pointers.current.delete(event.pointerId);
    pinchRef.current = 0;
    if (pointers.current.size > 0) return;
    dragRef.current = null;
    if (!drag) return;
    if (!drag.moved && stageRef.current) {
      const point = localPoint(event);
      const id = stageRef.current.pick(point.x, point.y);
      focusRef.current = id && focusRef.current !== id ? id : null;
      hoverRef.current = id;
      setHover(id ? { id, x: point.x, y: point.y } : null);
      motionRef.current = restingMotion(motionRef.current.orbit);
    } else if (event.timeStamp - drag.lastTime > 80) {
      motionRef.current = restingMotion(motionRef.current.orbit);
    }
    requestRef.current();
  }

  function onKeyDown(event: React.KeyboardEvent) {
    const step = 0.15;
    const moves: Record<string, (current: Orbit) => Orbit> = {
      ArrowLeft: (current) => rotateOrbit(current, -step, 0),
      ArrowRight: (current) => rotateOrbit(current, step, 0),
      ArrowUp: (current) => rotateOrbit(current, 0, -step),
      ArrowDown: (current) => rotateOrbit(current, 0, step),
      "+": (current) => zoomOrbit(current, 0.85),
      "=": (current) => zoomOrbit(current, 0.85),
      "-": (current) => zoomOrbit(current, 1 / 0.85),
    };
    if (event.key === "0" || event.key === "Escape") {
      event.preventDefault();
      resetView();
      return;
    }
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    turn(move);
  }

  const hovered = hover ? scene.nodes.find((node) => node.id === hover.id) : undefined;
  const described = scene.nodes.filter((node) => node.label && node.detail);
  const ready = status === "ready";

  return (
    <div className={variant === "lab" ? "scene3d" : `scene3d scene3d-${variant}`}>
      <div
        ref={frameRef}
        className="scene3d-frame"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => {
          if (dragRef.current) return;
          hoverRef.current = null;
          setHover(null);
        }}
      >
        <canvas
          key={generation}
          ref={canvasRef}
          className="scene3d-canvas"
          role="img"
          tabIndex={0}
          aria-label={`3D view. ${scene.caption} Drag, or use the arrow keys, to rotate; plus and minus to zoom; zero to reset.`}
          onKeyDown={onKeyDown}
          hidden={status === "unsupported" || status === "lost"}
        />
        <div ref={labelLayerRef} className="scene3d-labels" aria-hidden="true" />
        {overlay}
        {ready && hovered && (hovered.label || hovered.detail) ? (
          <div className="scene3d-tip" aria-hidden="true" style={{ left: Math.min(hover!.x + 14, sizeRef.current.width - 250), top: hover!.y + 16 }}>
            {hovered.label ? <strong>{hovered.label}</strong> : null}
            {hovered.detail ? <span>{hovered.detail}</span> : null}
          </div>
        ) : null}
        {status === "unsupported" ? <p className="scene3d-notice">The 3D view needs WebGL, which this browser has turned off or does not support. The same state is shown below as text.</p> : null}
        {status === "lost" ? (
          <div className="scene3d-notice">
            <p>The browser stopped the 3D view to save graphics memory. The same state is shown below as text.</p>
            <button onClick={() => setGeneration((current) => current + 1)}>Reload 3D view</button>
          </div>
        ) : null}
        {status === "loading" ? <p className="scene3d-notice">Loading 3D view…</p> : null}
        {ready ? (
          <p className="scene3d-hint" aria-hidden="true">
            <span className="hint-mouse">Drag to rotate · Ctrl + scroll to zoom · click an object to centre it</span>
            <span className="hint-touch">Drag to rotate · pinch to zoom · tap an object to centre it</span>
          </p>
        ) : null}
      </div>
      <div className="scene3d-bar">
        <div className="scene3d-text">
          <p className="scene3d-caption">{scene.caption}</p>
          {scene.legend && scene.legend.length > 0 ? (
            <ul className="scene3d-legend" aria-label="Colours in the 3D view">
              {scene.legend.map((entry) => (
                <li key={`${entry.tone}-${entry.text}`}><span className="scene3d-swatch" style={{ background: `var(${TONE_TOKENS[entry.tone]})` }} />{entry.text}</li>
              ))}
            </ul>
          ) : null}
        </div>
        {ready ? (
          <div className="playback" role="group" aria-label="3D view controls">
            <button onClick={() => turn((orbit) => zoomOrbit(orbit, 0.85))}>Zoom in</button>
            <button onClick={() => turn((orbit) => zoomOrbit(orbit, 1 / 0.85))}>Zoom out</button>
            <button onClick={resetView}>Reset view</button>
            <button onClick={() => motionSetting.set(motion === "on" ? "off" : "on")}>{motion === "on" ? "Pause motion" : "Play motion"}</button>
            <button onClick={() => effectsSetting.set(NEXT_EFFECTS[effects])} title="Glow and soft shadows. Auto turns them off on slow graphics hardware.">Effects: {effects}</button>
          </div>
        ) : null}
      </div>
      {described.length > 0 ? (
        <ul className="sr-only" aria-label="Objects in the 3D view">
          {described.map((node) => <li key={node.id}>{node.label}: {node.detail}</li>)}
        </ul>
      ) : null}
    </div>
  );
}
