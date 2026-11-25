#!/usr/bin/env node
// Records the manual's animations as GIFs for the repository README.
//
//   npm run build && npm run start            (in another terminal)
//   npx playwright-core install chromium      (once)
//   npm run record -- [--only id,group] [--list] [--verbose] [--frames dir] [--base http://127.0.0.1:3000] [--out dir]
//
// The page's clock is replaced by one the recorder steps by hand, so every frame is taken a fixed
// time after the one before, however slowly the machine draws. Nothing here runs in the manual itself.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { PNG } from "pngjs";
import { createGifWriter, paletteFrom } from "./lib/gif.mjs";
import { plainLabel } from "./lib/readme.mjs";
import { RECORDINGS } from "./recordings.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(`--${name}`);
  return at >= 0 && args[at + 1] ? args[at + 1] : fallback;
};
const BASE = option("base", process.env.XURCROH_MANUAL_URL ?? "http://127.0.0.1:3000");
const OUT = path.resolve(option("out", path.join(here, "../../../docs/media/animations")));
const ONLY = option("only", "").split(",").filter(Boolean);
const VERBOSE = args.includes("--verbose");
const KEEP = option("frames", "");
const HEADER = 76;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Runs in the page before any of its own scripts. Animation frames go through the recorder from the
 * start: until the clock is stopped they are passed on as usual, and from then on they are handed out
 * once per recorded frame instead of sixty times a second, so the 3D scene is drawn once for every
 * picture taken, which is all a recording can show anyway. Owning the queue from the start means a
 * frame that was asked for before the stop is carried over, not left behind with the old clock.
 */
function installFrames() {
  const request = window.requestAnimationFrame.bind(window);
  const cancel = window.cancelAnimationFrame.bind(window);
  const waiting = new Map();
  let handles = 0;
  let held = false;
  window.requestAnimationFrame = (callback) => {
    handles += 1;
    const handle = handles;
    const entry = { callback, passed: 0 };
    waiting.set(handle, entry);
    if (!held) {
      entry.passed = request((time) => {
        if (waiting.get(handle) !== entry) return;
        waiting.delete(handle);
        callback(time);
      });
    }
    return handle;
  };
  window.cancelAnimationFrame = (handle) => {
    const entry = waiting.get(handle);
    if (!entry) return;
    waiting.delete(handle);
    if (entry.passed) cancel(entry.passed);
  };
  window.__recordHold = () => {
    held = true;
    for (const entry of waiting.values()) {
      if (entry.passed) cancel(entry.passed);
      entry.passed = 0;
    }
  };
  /** Runs every frame that was asked for; answers how many there were. */
  window.__recordFrame = () => {
    const due = [...waiting.values()];
    waiting.clear();
    const now = performance.now();
    for (const entry of due) entry.callback(now);
    return due.length;
  };
}

/** Runs in the page once its clock is stopped: from here CSS and SVG animations are stepped by the same amount as the clock. */
function installStepper() {
  if (typeof window.__recordHold !== "function") throw new Error("the recorder's frame queue is not in the page");
  window.__recordHold();
  window.__recordStep = (ms) => {
    for (const animation of document.getAnimations()) {
      try {
        if (animation.playState === "running") animation.pause();
        const now = Number(animation.currentTime) || 0;
        const end = animation.effect?.getComputedTiming().endTime;
        if (Number.isFinite(end) && now + ms >= Number(end)) animation.finish();
        else animation.currentTime = now + ms;
      } catch { /* an animation that ended in between */ }
    }
    for (const svg of document.querySelectorAll("svg")) {
      try {
        svg.pauseAnimations();
        svg.setCurrentTime(svg.getCurrentTime() + ms / 1000);
      } catch { /* an svg without a timeline */ }
    }
  };
  const style = document.createElement("style");
  style.textContent = "[data-recording-active] { outline: 3px solid #ffd166 !important; outline-offset: 3px !important; }";
  document.head.append(style);
}

/** One recording in progress: the helpers a recording's `run` function uses. */
class Session {
  constructor(page, spec) {
    this.page = page;
    this.spec = spec;
    this.frames = [];
    this.controls = [];
    this.notes = [];
    this.stepMs = spec.stepMs ?? 120;
    this.clip = null;
    this.marked = 0;
    this.spent = { clock: 0, shot: 0 };
    this.scrollTop = null;
    this.anchor = null;
    this.drawn = [];
  }

  /** Lets real time pass so that work the fake clock does not drive (rendering, WebCrypto) can finish. */
  async settle(ms = 12) {
    await sleep(ms);
    await this.page.evaluate(() => new Promise((resolve) => {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => resolve(null);
      channel.port2.postMessage(0);
    }));
  }

  /** Moves the page's time forward without recording. */
  async skip(seconds) {
    const steps = Math.round((seconds * 1000) / this.stepMs);
    for (let index = 0; index < steps; index += 1) await this.tick();
  }

  async tick() {
    const started = Date.now();
    await this.page.clock.runFor(this.stepMs);
    await this.settle();
    // Pressing a control can scroll the page; the recorded area is put back where it was first.
    const drawn = await this.page.evaluate(([ms, top, anchor]) => {
      if (anchor) {
        // The recorded part is kept where it was first measured, whether the page was scrolled or
        // something above it changed height.
        const tops = anchor.selectors.map((selector) => document.querySelector(selector)?.getBoundingClientRect().top).filter((value) => value !== undefined);
        const shift = tops.length > 0 ? Math.round(Math.min(...tops) - anchor.top) : 0;
        if (shift !== 0) window.scrollBy({ top: shift, behavior: "instant" });
      } else if (top !== null && window.scrollY !== top) {
        window.scrollTo({ top, behavior: "instant" });
      }
      const count = window.__recordFrame();
      window.__recordStep(ms);
      return count;
    }, [this.stepMs, this.scrollTop, this.anchor]);
    this.drawn.push(drawn);
    this.spent.clock += Date.now() - started;
  }

  /** Records `seconds` of page time. */
  async record(seconds) {
    const steps = Math.round((seconds * 1000) / this.stepMs);
    for (let index = 0; index < steps; index += 1) {
      await this.tick();
      const started = Date.now();
      this.frames.push(await this.page.screenshot({ clip: this.clip, type: "png", timeout: 120_000 }));
      this.spent.shot += Date.now() - started;
      if (VERBOSE && this.frames.length % 10 === 0) console.error(`  ${this.spec.id}: ${this.frames.length} frames; per frame ${Math.round(this.spent.clock / this.frames.length)} ms clock, ${Math.round(this.spent.shot / this.frames.length)} ms picture`);
      if (this.marked > 0) {
        this.marked -= 1;
        if (this.marked === 0) await this.page.evaluate(() => document.querySelectorAll("[data-recording-active]").forEach((element) => element.removeAttribute("data-recording-active")));
      }
    }
  }

  /** Records until `test` (run in the page) is true, for at most `maxSeconds`; then `after` seconds more. */
  async recordUntil(test, maxSeconds, after = 0, arg = undefined) {
    const limit = Math.round((maxSeconds * 1000) / this.stepMs);
    for (let index = 0; index < limit; index += 1) {
      await this.record(this.stepMs / 1000);
      if (await this.page.evaluate(test, arg)) { await this.record(after); return true; }
    }
    this.notes.push(`stopped at the ${maxSeconds} s limit`);
    return false;
  }

  /** Moves time forward, unrecorded, until `test` is true. */
  async skipUntil(test, maxSeconds, arg = undefined) {
    const limit = Math.round((maxSeconds * 1000) / this.stepMs);
    for (let index = 0; index < limit; index += 1) {
      if (await this.page.evaluate(test, arg)) { await this.unmark(); return true; }
      await this.tick();
    }
    throw new Error(`${this.spec.id}: waited ${maxSeconds} s of page time for a state that did not come`);
  }

  /** Takes the outline off again: the time in which it would have shown went by unrecorded. */
  async unmark() {
    this.marked = 0;
    await this.page.evaluate(() => document.querySelectorAll("[data-recording-active]").forEach((element) => element.removeAttribute("data-recording-active")));
  }

  /** Outlines a control for the next few recorded frames, standing in for a visible mouse pointer. */
  async mark(locator) {
    await this.page.evaluate(() => document.querySelectorAll("[data-recording-active]").forEach((element) => element.removeAttribute("data-recording-active")));
    await locator.evaluate((element) => (element.closest("label") ?? element).setAttribute("data-recording-active", ""));
    this.marked = Math.max(2, Math.round(700 / this.stepMs));
  }

  /** Writes down a control this recording shows being used. */
  used(label) {
    const tidy = label.replace(/\s+/g, " ").trim();
    if (tidy && !this.controls.includes(tidy)) this.controls.push(tidy);
  }

  button(name, scope) {
    return (scope ? this.page.locator(scope) : this.page).getByRole("button", { name, exact: typeof name === "string" }).first();
  }

  /** Presses a button by its visible name, as a reader would. */
  async click(name, { scope, as } = {}) {
    const target = this.button(name, scope);
    await this.mark(target);
    await target.click({ timeout: 20_000 });
    this.used(as ?? (typeof name === "string" ? name : await target.innerText()));
    await this.settle(30);
  }

  /** Chooses an option of a labelled drop-down. */
  async select(label, value, { scope, as } = {}) {
    const target = (scope ? this.page.locator(scope) : this.page).getByLabel(label).first();
    await this.mark(target);
    await target.selectOption(value);
    this.used(as ?? `${label}: ${typeof value === "string" ? value : value.label ?? value.value}`);
    await this.settle(30);
  }

  /** Hides everything matched by `css`, so that only the part being recorded is drawn (and computed). */
  async hide(css) {
    await this.page.addStyleTag({ content: `${css} { display: none !important; }` });
    await this.settle(400);
  }

  /** Records the whole window instead of one part of the page. */
  async frameWindow() {
    const viewport = this.page.viewportSize();
    await this.page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
    await this.settle(250);
    this.clip = { x: 0, y: 0, width: viewport.width, height: viewport.height };
    this.scrollTop = 0;
    this.anchor = null;
  }

  /** Types into a labelled field one character at a time, recording as it goes. */
  async type(label, text, { scope, perKey = 0.12 } = {}) {
    const target = (scope ? this.page.locator(scope) : this.page).getByLabel(label).first();
    await this.mark(target);
    await target.fill("");
    for (const character of text) {
      await target.pressSequentially(character);
      await this.settle(20);
      await this.record(perKey);
    }
    this.used(label);
  }

  /** Drags across an element (fractions of its box), recording while the pointer moves. */
  async drag(selector, from, to, { steps = 8, as } = {}) {
    const box = await this.page.locator(selector).first().boundingBox();
    const at = (fraction) => ({ x: box.x + box.width * (from[0] + (to[0] - from[0]) * fraction), y: box.y + box.height * (from[1] + (to[1] - from[1]) * fraction) });
    const start = at(0);
    await this.page.mouse.move(start.x, start.y);
    await this.page.mouse.down();
    for (let index = 1; index <= steps; index += 1) {
      const point = at(index / steps);
      await this.page.mouse.move(point.x, point.y);
      await this.settle(20);
      await this.record(this.stepMs / 1000);
    }
    await this.page.mouse.up();
    if (as) this.used(as);
    await this.settle(30);
  }

  /** The visible names of the buttons in a labelled group. */
  async names(group, scope) {
    return (scope ? this.page.locator(scope) : this.page).getByRole("group", { name: group, exact: true }).first().getByRole("button").allInnerTexts();
  }

  /**
   * Uses every control of one laboratory in turn: its parameters, its attack, its playback buttons and
   * its header buttons. Each is changed, shown for a moment and put back. Returns nothing; the labels
   * are written down as it goes.
   */
  async labControls(id, { dwell = 1.1 } = {}) {
    const lab = `#${id}`;
    const describe = () => this.page.evaluate((selector) => {
      const own = (element) => [...element.childNodes].filter((node) => node.nodeType === 3).map((node) => node.textContent).join(" ").replace(/\s+/g, " ").trim();
      const controls = [];
      const root = document.querySelector(`${selector} .lab-controls`);
      if (!root) throw new Error(`${selector} has no .lab-controls`);
      root.querySelectorAll("input, select, button").forEach((element, index) => {
        element.setAttribute("data-recording-control", String(index));
        const label = element.closest("label");
        const playback = Boolean(element.closest(".playback"));
        const kind = element.tagName === "SELECT" ? "select" : element.tagName === "BUTTON" ? "button" : element.type;
        controls.push({
          index, kind, playback,
          label: label ? own(label) || label.textContent.trim() : element.textContent.trim(),
          value: element.value ?? "", min: element.min ?? "", max: element.max ?? "", step: element.step ?? "", checked: Boolean(element.checked),
          options: kind === "select" ? [...element.options].map((entry) => ({ value: entry.value, text: entry.textContent.trim() })) : [],
        });
      });
      return controls;
    }, lab);
    const at = (control) => this.page.locator(`${lab} [data-recording-control="${control.index}"]`);
    const plain = plainLabel;

    const controls = await describe();
    // The lab's own parameters first; then its playback buttons, in an order in which each can be pressed.
    const order = ["Step forward", "Step back", "Reset", "Play", "Pause"];
    const rank = (entry) => (entry.kind === "button" && order.includes(entry.label) ? order.indexOf(entry.label) : order.length);
    const playback = controls.filter((entry) => entry.playback).sort((a, b) => rank(a) - rank(b));
    for (const control of [...controls.filter((entry) => !entry.playback), ...playback]) {
      const target = at(control);
      const name = plain(control.label);
      if (control.kind === "checkbox") {
        await this.mark(target);
        await target.click();
        this.used(name);
        await this.record(dwell * 1.4);
        await target.click();
        await this.record(dwell * 0.5);
      } else if (control.kind === "select") {
        for (const entry of control.options.filter((item) => item.value !== control.value)) {
          await this.mark(target);
          await target.selectOption(entry.value);
          this.used(`${name}: ${entry.text}`);
          await this.record(dwell);
        }
        await target.selectOption(control.value);
        this.used(`${name}: ${control.options.find((item) => item.value === control.value)?.text ?? control.value}`);
        await this.record(dwell * 0.5);
      } else if (control.kind === "range" || control.kind === "number") {
        const low = Number(control.min || 0);
        const high = Number(control.max || 10);
        const original = Number(control.value);
        const values = control.kind === "range" ? [low, high] : [Math.min(high, original + 1), Math.min(high, original + 3)];
        // A slider is set the way dragging it would: the browser moves it to the nearest allowed position
        // (its ends need not be whole steps), and the page hears about it as it does from a reader.
        const put = control.kind === "range"
          ? (value) => target.evaluate((element, next) => {
            Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(element, String(next));
            element.dispatchEvent(new Event("input", { bubbles: true }));
            element.dispatchEvent(new Event("change", { bubbles: true }));
          }, value)
          : (value) => target.fill(String(value));
        for (const value of values.filter((entry) => entry !== original)) {
          await this.mark(target);
          await put(value);
          await this.record(dwell);
        }
        this.used(name);
        await put(original);
        await this.record(dwell * 0.5);
      } else if (control.kind === "text") {
        // Another valid value of the same kind: the same list, the other way round.
        await this.mark(target);
        await target.fill(control.value.split(",").map((part) => part.trim()).reverse().join(","));
        this.used(name);
        await this.record(dwell * 1.4);
        await target.fill(control.value);
        await this.record(dwell * 0.5);
      } else if (control.kind === "button") {
        await this.mark(target);
        const text = (await target.innerText()).trim();
        if (await target.isDisabled()) {
          // "Step back" at the first step: go one step on first.
          const forward = controls.find((entry) => entry.label === "Step forward");
          if (forward) { await at(forward).click(); await this.record(dwell * 0.5); }
        }
        if (await target.isDisabled()) { this.notes.push(`“${text}” was switched off at that moment`); continue; }
        await target.click();
        this.used(text);
        await this.record(dwell);
        // A button that changes its name is a switch: press it again so that both states are shown.
        const after = (await target.innerText()).trim();
        if (after !== text) {
          await this.mark(target);
          await target.click();
          this.used(after);
          await this.record(dwell);
        }
      }
    }

    // The attack button under the stage: on, then off again.
    const attack = this.page.locator(`${lab} .liveio-attack button`);
    if (await attack.count() > 0) {
      const text = (await attack.first().innerText()).trim();
      await this.mark(attack.first());
      await attack.first().click();
      this.used(text);
      await this.record(dwell * 2);
      await attack.first().click();
      await this.record(dwell * 0.6);
    }

    // The header: Watch / Stop, and the view without the 3D scene.
    for (const text of ["Stop", "Watch", "Stop", "Hide 3D view", "Show 3D view"]) {
      const target = this.page.locator(`${lab} .lab-header`).getByRole("button", { name: text, exact: true });
      if (await target.count() === 0) continue;
      await this.mark(target.first());
      await target.first().click();
      this.used(text);
      await this.record(text === "Hide 3D view" ? dwell * 2 : dwell);
    }
  }

  /** Scrolls so that `region` sits under the page header and fixes the recorded area; `extend` adds height below it. */
  async frame(region, { extend = 0 } = {}) {
    const parts = Array.isArray(region) ? region : [region];
    await this.page.locator(parts[0]).first().waitFor({ state: "visible", timeout: 60_000 });
    const measure = () => this.page.evaluate((selectors) => {
      let box = null;
      for (const selector of selectors) {
        const element = document.querySelector(selector);
        if (!element) throw new Error(`Nothing matches ${selector}`);
        const rect = element.getBoundingClientRect();
        box = box ? { left: Math.min(box.left, rect.left), top: Math.min(box.top, rect.top), right: Math.max(box.right, rect.right), bottom: Math.max(box.bottom, rect.bottom) } : { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
      }
      return { ...box, scrollY: window.scrollY };
    }, parts);
    const before = await measure();
    await this.page.evaluate((top) => window.scrollTo({ top, behavior: "instant" }), Math.max(0, Math.round(before.top + before.scrollY - HEADER)));
    await this.settle(250);
    const box = await measure();
    const pad = this.spec.pad ?? 0;
    const even = (value) => value - (value % 2);
    this.clip = { x: Math.max(0, Math.floor(box.left) - pad), y: Math.max(0, Math.floor(box.top) - pad), width: even(Math.ceil(box.right - box.left) + 2 * pad), height: even(Math.ceil(box.bottom - box.top) + 2 * pad + extend) };
    this.scrollTop = await this.page.evaluate(() => window.scrollY);
    this.anchor = { selectors: parts, top: box.top };
    const viewport = this.page.viewportSize();
    if (this.clip.y + this.clip.height > viewport.height) throw new Error(`${this.spec.id}: the recorded area (${this.clip.height}px from ${this.clip.y}px) does not fit a ${viewport.height}px window; raise its viewport height`);
  }

  /** Waits (in real time) for the 3D view inside `selector` to have drawn. */
  async ready(selector) {
    await this.page.locator(`${selector} canvas`).first().waitFor({ state: "attached", timeout: 90_000 });
    await this.page.locator(selector).getByRole("group", { name: "3D view controls" }).first().waitFor({ state: "visible", timeout: 90_000 });
  }

  /** Stops the page's own clock; from here on time only moves when the recorder says so. */
  async hold() {
    // The pause must lie in the page's future. Its clock keeps running while the request travels, and
    // catches up in jumps when the page was busy, so the margin is generous and widened if it was not enough.
    let target = 0;
    for (const ahead of [20_000, 120_000, 900_000]) {
      const now = await this.page.evaluate(() => Date.now());
      try {
        await this.page.clock.pauseAt(now + ahead);
        target = now + ahead;
        break;
      } catch (error) {
        if (!String(error).includes("to the past")) throw error;
      }
    }
    if (target === 0) throw new Error(`${this.spec.id}: could not stop the page's clock`);
    // The pause can arrive while the clock is in the middle of one of its own real-time steps. That step
    // then ends after the pause and sets the time back to where it had been, which leaves everything the
    // page scheduled during the pause (the next frame of a 3D view, for one) that far in the future: the
    // view would stand still for the first twenty seconds of the recording. Check, and bring the time
    // forward again until it stays.
    let at = 0;
    for (let attempt = 0; attempt < 6; attempt += 1) {
      await this.settle(80);
      at = await this.page.evaluate(() => Date.now());
      if (at >= target) break;
      if (VERBOSE) console.error(`  ${this.spec.id}: the clock had fallen back by ${target - at} ms; brought forward again`);
      await this.page.clock.fastForward(target - at);
    }
    if (at < target) throw new Error(`${this.spec.id}: the page's clock would not stay stopped at one time`);
    await this.page.evaluate(installStepper);
    await this.settle(50);
  }
}

async function encode(session, file) {
  const decoded = (png) => PNG.sync.read(png);
  const first = decoded(session.frames[0]);
  const every = Math.max(1, Math.floor(session.frames.length / 14));
  const samples = session.frames.filter((_, index) => index % every === 0).map((png) => decoded(png).data);
  const writer = createGifWriter({ width: first.width, height: first.height, palette: paletteFrom(samples, session.spec.colors ?? 96), fuzz: session.spec.fuzz ?? 24, settleEvery: session.spec.settleEvery ?? 8 });
  samples.length = 0;
  if (KEEP) {
    // For looking at single frames, or trying other encoder settings, without recording again.
    await mkdir(path.join(KEEP, session.spec.id), { recursive: true });
    await Promise.all(session.frames.map((png, at) => writeFile(path.join(KEEP, session.spec.id, `${String(at).padStart(4, "0")}.png`), png)));
  }
  for (const png of session.frames) writer.add(decoded(png).data, session.stepMs);
  const bytes = writer.finish();
  await writeFile(file, bytes);
  return { width: first.width, height: first.height, bytes: bytes.length, stored: writer.written };
}

async function main() {
  if (args.includes("--list")) {
    for (const spec of RECORDINGS) console.log(`${spec.id.padEnd(34)} ${spec.page.padEnd(26)} ${spec.title}`);
    return;
  }
  const chosen = RECORDINGS.filter((spec) => ONLY.length === 0 || ONLY.includes(spec.id) || ONLY.includes(spec.group));
  if (chosen.length === 0) throw new Error(`No recording matches ${ONLY.join(", ")}`);
  await mkdir(OUT, { recursive: true });
  const indexFile = path.join(OUT, "index.json");
  let index = {};
  try { index = JSON.parse(await readFile(indexFile, "utf8")); } catch { index = {}; }

  // Software rendering keeps the pictures the same on every machine. Putting the page together on the
  // processor rather than through the software graphics card makes each screenshot several times faster.
  const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--disable-gpu-compositing", "--no-sandbox", "--force-color-profile=srgb"] });
  let failed = 0;
  try {
    for (const spec of chosen) {
      const started = Date.now();
      const context = await browser.newContext({ viewport: spec.viewport ?? { width: 1440, height: 1000 }, deviceScaleFactor: spec.scale ?? 0.75, colorScheme: spec.theme === "dark" ? "dark" : "light", reducedMotion: "no-preference" });
      const page = await context.newPage();
      const problems = [];
      page.on("pageerror", (error) => problems.push(error.message));
      await page.addInitScript((settings) => {
        try { for (const [key, value] of Object.entries(settings)) window.localStorage.setItem(key, value); } catch { /* storage blocked */ }
      }, { "xurcroh-3d-effects": "on", ...(spec.theme ? { "xurcroh-theme": spec.theme } : {}), ...(spec.settings ?? {}) });
      const session = new Session(page, spec);
      try {
        await page.clock.install();
        // After the clock, so that the frames passed on before the stop follow the clock too.
        await page.addInitScript(installFrames);
        await page.goto(new URL(spec.page, BASE).href, { waitUntil: "networkidle", timeout: 120_000 });
        await spec.run(session);
        if (session.frames.length === 0) throw new Error("nothing was recorded");
        if (problems.length > 0) throw new Error(`the page reported errors: ${problems.join(" | ")}`);
        const result = await encode(session, path.join(OUT, spec.file));
        index[spec.id] = { file: spec.file, ...result, frames: session.frames.length, seconds: Number(((session.frames.length * session.stepMs) / 1000).toFixed(1)), controls: session.controls, notes: session.notes };
        await writeFile(indexFile, `${JSON.stringify(Object.fromEntries(Object.entries(index).sort(([a], [b]) => a.localeCompare(b))), null, 2)}\n`);
        if (VERBOSE) console.error(`  ${spec.id}: animation-frame callbacks per tick: ${session.drawn.join("")}`);
        console.log(`${spec.id}: ${result.width}×${result.height}, ${session.frames.length} frames (${result.stored} stored), ${(result.bytes / 1e6).toFixed(2)} MB, took ${Math.round((Date.now() - started) / 1000)} s`);
      } catch (error) {
        failed += 1;
        console.error(`${spec.id}: FAILED: ${error instanceof Error ? error.message.split("\n")[0] : error}`);
      } finally {
        await context.close();
      }
    }
  } finally {
    await browser.close();
  }
  if (failed > 0) process.exitCode = 1;
}

main().catch((error) => { console.error(error); process.exit(1); });
