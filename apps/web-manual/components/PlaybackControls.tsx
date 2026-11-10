"use client";

export interface PlaybackProps {
  playing: boolean;
  onToggle: () => void;
  onStep: (delta: number) => void;
  onReset: () => void;
  speed: number;
  onSpeed: (speed: number) => void;
  canStepBackward: boolean;
}

export function PlaybackControls({ playing, onToggle, onStep, onReset, speed, onSpeed, canStepBackward }: PlaybackProps) {
  return (
    <div className="playback" role="group" aria-label="Animation controls">
      <button onClick={onToggle}>{playing ? "Pause" : "Play"}</button>
      <button onClick={() => onStep(-1)} disabled={!canStepBackward}>Step back</button>
      <button onClick={() => onStep(1)}>Step forward</button>
      <button onClick={onReset}>Reset</button>
      <label>Speed
        <select value={speed} onChange={(event) => onSpeed(Number(event.target.value))}>
          <option value={0.5}>0.5×</option>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
          <option value={4}>4×</option>
        </select>
      </label>
    </div>
  );
}
