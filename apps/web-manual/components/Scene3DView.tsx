"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_ORBIT, rotateOrbit, zoomOrbit, type Orbit } from "@/lib/scene/orbit";
import type { LabelPoint, Palette, Stage } from "@/lib/scene/stage";
import type { Scene3D } from "@/lib/scene/types";

const FALLBACK: Palette = { plain: "#8a99a8", accent: "#075985", good: "#15803d", warn: "#9a3412", muted: "#c3ccd6", text: "#17212b", grid: "#ccd5de" };

function readPalette(): Palette {
  const style = getComputedStyle(document.documentElement);
  const token = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    plain: token("--muted", FALLBACK.plain),
    accent: token("--accent", FALLBACK.accent),
    good: token("--good", FALLBACK.good),
    warn: token("--warning", FALLBACK.warn),
    muted: token("--border", FALLBACK.muted),
    text: token("--text", FALLBACK.text),
    grid: token("--border", FALLBACK.grid),
  };
}

/** Interactive 3D drawing of a lab's current state. The lab's own text and tables remain the exact view. */
export function Scene3DView({ scene }: { scene: Scene3D }) {
  const frameRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<Stage | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "unsupported">("loading");
  const [orbit, setOrbit] = useState<Orbit>(DEFAULT_ORBIT);
  const [labels, setLabels] = useState<LabelPoint[]>([]);
  const [size, setSize] = useState({ width: 640, height: 340 });
  const [themeTick, setThemeTick] = useState(0);
  // Labs rebuild their scene object on every render; redraw only when its content changes.
  const sceneKey = useMemo(() => JSON.stringify(scene), [scene]);
  const sceneRef = useRef(scene);
  sceneRef.current = scene;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (typeof WebGLRenderingContext === "undefined") {
      setStatus("unsupported");
      return;
    }
    let cancelled = false;
    let stage: Stage | null = null;
    void import("@/lib/scene/stage").then(({ createStage }) => {
      if (cancelled) return;
      stage = createStage(canvas);
      stageRef.current = stage;
      setStatus("ready");
    }).catch(() => {
      if (!cancelled) setStatus("unsupported");
    });
    return () => {
      cancelled = true;
      stage?.dispose();
      stageRef.current = null;
    };
  }, []);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;
    const measure = () => {
      const box = frame.getBoundingClientRect();
      if (box.width > 0 && box.height > 0) setSize({ width: Math.round(box.width), height: Math.round(box.height) });
    };
    measure();
    const resize = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    resize?.observe(frame);
    const theme = new MutationObserver(() => setThemeTick((tick) => tick + 1));
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      resize?.disconnect();
      theme.disconnect();
    };
  }, []);

  useEffect(() => {
    if (status !== "ready") return;
    stageRef.current?.update(sceneRef.current, readPalette());
  }, [status, sceneKey, themeTick]);

  useEffect(() => {
    if (status !== "ready" || !stageRef.current) return;
    setLabels(stageRef.current.view(orbit, size.width, size.height));
  }, [status, sceneKey, themeTick, orbit, size]);

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
      "0": () => DEFAULT_ORBIT,
    };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    setOrbit(move);
  }

  return (
    <div className="scene3d">
      <div
        ref={frameRef}
        className="scene3d-frame"
        onPointerDown={(event) => {
          dragRef.current = { x: event.clientX, y: event.clientY };
          event.currentTarget.setPointerCapture?.(event.pointerId);
        }}
        onPointerMove={(event) => {
          const last = dragRef.current;
          if (!last) return;
          const deltaX = event.clientX - last.x;
          const deltaY = event.clientY - last.y;
          dragRef.current = { x: event.clientX, y: event.clientY };
          setOrbit((current) => rotateOrbit(current, -deltaX * 0.01, -deltaY * 0.01));
        }}
        onPointerUp={() => { dragRef.current = null; }}
        onPointerCancel={() => { dragRef.current = null; }}
      >
        <canvas
          ref={canvasRef}
          className="scene3d-canvas"
          role="img"
          tabIndex={0}
          aria-label={`3D view. ${scene.caption} Drag, or use the arrow keys, to rotate; plus and minus to zoom; zero to reset.`}
          onKeyDown={onKeyDown}
          hidden={status === "unsupported"}
        />
        {status === "ready" ? (
          <div className="scene3d-labels" aria-hidden="true">
            {labels.filter((label) => label.visible).map((label) => (
              <span key={label.id} className={label.active ? "scene3d-label active" : "scene3d-label"} style={{ left: label.x, top: label.y }}>{label.text}</span>
            ))}
          </div>
        ) : null}
        {status === "unsupported" ? <p className="scene3d-notice">The 3D view needs WebGL, which this browser has turned off or does not support. The same state is shown below as text.</p> : null}
        {status === "loading" ? <p className="scene3d-notice">Loading 3D view…</p> : null}
      </div>
      <div className="scene3d-bar">
        <p className="small scene3d-caption">{scene.caption}</p>
        {status === "ready" ? (
          <div className="playback" role="group" aria-label="3D view controls">
            <button onClick={() => setOrbit((current) => zoomOrbit(current, 0.85))}>Zoom in</button>
            <button onClick={() => setOrbit((current) => zoomOrbit(current, 1 / 0.85))}>Zoom out</button>
            <button onClick={() => setOrbit(DEFAULT_ORBIT)}>Reset view</button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
