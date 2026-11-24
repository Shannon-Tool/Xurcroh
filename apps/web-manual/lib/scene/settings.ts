import type { EffectsPreference } from "@/lib/scene/quality";

/** A per-browser choice shared by every 3D view on the page. Storage may be unavailable; the page works without it. */
export interface Setting<T extends string> {
  get(): T;
  set(value: T): void;
  subscribe(listener: () => void): () => void;
}

function createSetting<T extends string>(key: string, fallback: T, allowed: readonly T[]): Setting<T> {
  const listeners = new Set<() => void>();
  let cached: T | null = null;
  return {
    get() {
      if (cached !== null) return cached;
      let stored: string | null = null;
      try {
        stored = window.localStorage.getItem(key);
      } catch {
        stored = null;
      }
      cached = allowed.includes(stored as T) ? (stored as T) : fallback;
      return cached;
    },
    set(value) {
      cached = value;
      try {
        window.localStorage.setItem(key, value);
      } catch {
        // Private windows and blocked storage: the choice lasts for this page only.
      }
      listeners.forEach((listener) => listener());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
  };
}

export const effectsSetting = createSetting<EffectsPreference>("xurcroh-3d-effects", "auto", ["auto", "on", "off"]);
export const motionSetting = createSetting<"on" | "off">("xurcroh-3d-motion", "on", ["on", "off"]);
/** Plain everyday wording, or the technical terms the chapters use. */
export const wordsSetting = createSetting<"plain" | "technical">("xurcroh-words", "plain", ["plain", "technical"]);
