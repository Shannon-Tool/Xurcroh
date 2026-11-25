// The list of recordings shown in the repository README: what each one is, where it is taken,
// and what the recorder does while it runs. `record-animations.mjs` plays them; `render-readme.mjs`
// writes the README section from them.

const STAGE = ".theater .scene3d-frame";
const PIPELINE_CONTROLS = [STAGE, ".show-attacks"];

/** True once the pipeline's run has produced everything (its "Next step" button switches off). */
const pipelineDone = () => document.querySelector(".show-controls button:nth-of-type(2)")?.disabled === true;
/** True once the pipeline has produced its first unit and is streaming the rest. */
const pipelineStreaming = () => Number(document.querySelector('.theater [role="progressbar"]')?.getAttribute("aria-valuenow") ?? 0) > 0;
const rejections = () => Number(document.querySelector('[data-testid="rejected"]')?.textContent ?? 0);

/** Opens the pipeline on its stage and starts a run from the first step. */
async function startPipeline(r, region = STAGE) {
  await r.ready(".theater");
  await r.frame(region);
  await r.hold();
  await r.click("Restart");
}

/** Opens the pipeline and moves on, unrecorded, to where it streams one unit after another. */
async function streamingPipeline(r, region = STAGE) {
  await startPipeline(r, region);
  await r.skipUntil(pipelineStreaming, 20);
  await r.skip(0.6);
}

/** Opens one laboratory on the laboratory page, alone, with its 3D view drawn. */
async function openLab(r, id) {
  // The other laboratories are not drawn, and the page's own index bar, which would cover the top of the stage, is left out.
  await r.hide(`section.lab:not(#${id}), .lab-nav`);
  // Taking the other laboratories away moves this one; its 3D view only starts once it is on screen.
  await r.page.locator(`#${id} .scene3d-frame`).evaluate((element) => element.scrollIntoView({ block: "center", behavior: "instant" }));
  await r.ready(`#${id}`);
}

const LABS = [
  { id: "share-grid", title: "Splitting your question", says: "Your question becomes numbers, cut into two random pieces; four workers each work on one piece." },
  { id: "freivalds", title: "Catching a wrong answer", says: "Every result a worker returns is tested against a random challenge before it is used." },
  { id: "quantized-formats", title: "Storing the model in pieces", says: "The model's compressed numbers are decoded exactly and split into pieces as well." },
  { id: "placement", title: "Deciding who gets which piece", says: "Pieces that fit together are placed with different providers in different countries." },
  { id: "conjugation", title: "Disguising the numbers", says: "The numbers are rotated with a secret rotation before they leave. A disguise, not encryption." },
  { id: "fwht", title: "Mixing the numbers quickly", says: "A fast mixing transform spreads every number across all the others." },
  { id: "pipeline", title: "Getting your answer when a worker fails", says: "Requests pass from worker to worker; a failed worker is noticed and the work is sent again." },
  { id: "onion", title: "Hiding the route", says: "The message is wrapped in layers; each relay removes one and learns only the next stop." },
  { id: "moe", title: "Hiding which experts were used", says: "Every expert receives the same amount of work, so the load does not reveal the question." },
  { id: "diffusion", title: "Making a picture", says: "A picture is produced step by step from numbers the worker only sees in disguise." },
  { id: "memory", title: "Wiping memory afterwards", says: "Memory is wiped, and the wipe confirmed finished, before it is handed to anyone else." },
];

const labRecordings = LABS.flatMap((lab) => [
  {
    id: `lab-${lab.id}`,
    group: "labs",
    file: `lab-${lab.id}.gif`,
    title: lab.title,
    caption: `${lab.says} The cards along the bottom show the real values of each step; the answer is written only after the last one.`,
    page: `/laboratory#${lab.id}`,
    // The gallery lists this lab's stage under `lab-pipeline`, because `pipeline` is the live demonstration.
    stage: lab.id === "pipeline" ? "lab-pipeline" : lab.id,
    async run(r) {
      await openLab(r, lab.id);
      await r.frame(`#${lab.id} .scene3d-frame`);
      await r.hold();
      const steps = r.page.locator(`#${lab.id}`).getByRole("button", { name: "Step back", exact: true });
      if (await steps.count() === 0) {
        // No steps to play: the scene turns and the answer is typed out again and again.
        await r.record(8);
        return;
      }
      // One whole loop, from the first step until it comes round to the first step again.
      await r.page.locator(`#${lab.id}`).getByRole("button", { name: "Reset", exact: true }).click();
      await r.page.locator(`#${lab.id}`).getByRole("button", { name: "Watch", exact: true }).click();
      await r.settle(40);
      await r.record(0.4);
      await r.recordUntil((id) => document.querySelector(`#${id} .lab-controls .playback button:nth-of-type(2)`)?.disabled === false, 6, 0, lab.id);
      // A loop now ends with the answer being written, so it lasts longer than the steps alone.
      await r.recordUntil((id) => document.querySelector(`#${id} .lab-controls .playback button:nth-of-type(2)`)?.disabled === true, 44, 1.2, lab.id);
    },
  },
  {
    id: `lab-${lab.id}-controls`,
    group: "lab-controls",
    file: `lab-${lab.id}-controls.gif`,
    title: `${lab.title}: every control`,
    caption: "Each control is changed in turn and put back.",
    page: `/laboratory#${lab.id}`,
    viewport: { width: 1440, height: 1500 },
    scale: 0.7,
    async run(r) {
      await openLab(r, lab.id);
      // Only the stage and the controls are kept in the picture; the text between them is left out.
      await r.hide(`#${lab.id} .lab-header > div:first-child, #${lab.id} .liveio-facts, #${lab.id} .liveio-honest, #${lab.id} .strength, #${lab.id} .lab-exact > :not(summary):not(.lab-controls)`);
      await r.frame(`#${lab.id}`);
      await r.hold();
      // The 3D scene stands still here so that only the controls' effects change the picture.
      await r.click("Pause motion", { scope: `#${lab.id}` });
      await r.record(0.6);
      await r.labControls(lab.id);
      await r.record(0.6);
    },
  },
]);

const EYES = [
  { id: "worker", label: "One worker", says: "One worker's view: one piece of your data and one piece of the model. Noise." },
  { id: "same", label: "Two workers, same piece", says: "Two workers that hold the same piece of your data, pooling what they have. Still noise." },
  { id: "match", label: "Two workers, matching pieces", says: "Two workers that hold both pieces rebuild what you see. This is where the design stops: it relies on such workers not pooling what they hold." },
  { id: "relay", label: "A Tor relay", says: "The middle relay's view: sealed bytes and the next stop." },
  { id: "wire", label: "The network", says: "Someone watching the network sees encrypted bytes: how much and when, not what." },
  { id: "operator", label: "The machine's operator", says: "The operator of the machine sees an enclave it cannot look into (simulated here; one more layer, not proof)." },
];

const ATTACKS = [
  { id: "result", label: "Tamper with a worker", says: "One worker's result is changed by one. The check rejects it and the step is done again." },
  { id: "software", label: "Swap a worker's software", says: "A worker's signed report no longer matches the expected software. Nothing is sent to it." },
  { id: "packet", label: "Alter a sealed packet", says: "One bit of a sealed packet is flipped. The relay's authenticity check fails and the packet is dropped." },
];

const DIAGRAMS = [
  { id: "architecture", page: "/05-system-architecture", title: "System architecture", says: "From the question on your device to the combined answer, part by part." },
  { id: "runtime", page: "/07-inference-runtime", title: "Runtime loop", says: "One token round the loop: split, compute, check, combine, next layer, next token." },
  { id: "memory", page: "/09-memory-security", title: "Memory life cycle", says: "Allocate, execute, release, synchronise, zero, reuse, and what goes wrong if a step is skipped." },
];

export const GROUPS = [
  { id: "pipeline", title: "The pipeline", lead: "A question goes in on the left and the answer comes out on the right. In between it is shuffled, split into random pieces, sealed in three layers, passed through three relays, computed by four workers in enclaves, checked and put back together." },
  { id: "eyes", title: "The pipeline through someone else's eyes", lead: "“Whose eyes?” rebuilds the view of another party from what that party really holds." },
  { id: "attacks", title: "Trying to break the pipeline", lead: "Each attack is caught by the real check, and the step is done again." },
  { id: "strength", title: "How hard it is to undo", lead: "Facts the page computes about recovering what was hidden, each with the place where it stops. No ratings and no adjectives." },
  { id: "labs", title: "The eleven laboratories", lead: "Each laboratory runs one real computation from what goes in to what comes out. The cards along the bottom are its steps, each with the actual values of that step; the lit part of the 3D scene carries the same values; the output panel stays empty until the last step is done, and then the answer is written one token at a time." },
  { id: "lab-controls", title: "Every control of every laboratory", lead: "The same laboratories with their controls underneath. Every step card is pressed first, then every control. A yellow outline marks the control being used; the 3D scene is held still so that only the control's effect changes the picture, and the text between the stage and the controls is left out of the frame." },
  { id: "tour", title: "The protection tour", lead: "The protection as seven layers around the data, one at a time." },
  { id: "diagrams", title: "Chapter diagrams", lead: "The schematic drawings of chapters 5, 7 and 9, stepped through with the real values of the chosen question." },
  { id: "settings", title: "Settings", lead: "The choices a reader can make on any page." },
];

export const RECORDINGS = [
  {
    id: "pipeline-text",
    group: "pipeline",
    lead: true,
    file: "pipeline-text.gif",
    title: "Text: a prepared question to its answer",
    caption: "All eight steps once, slowly, then the rest of the answer token by token.",
    page: "/show",
    stage: "pipeline",
    // The longest recording: a few colours fewer keep it under the limit for one file.
    colors: 80,
    settleEvery: 10,
    async run(r) {
      await startPipeline(r);
      // Shown without being pressed: text is what the demonstration opens with, seen through the reader's own eyes.
      r.used("Text");
      r.used("You");
      await r.recordUntil(pipelineDone, 40, 2.4);
    },
  },
  {
    id: "pipeline-picture",
    group: "pipeline",
    file: "pipeline-picture.gif",
    title: "Picture",
    caption: "A prepared picture forms step by step on your device; the workers' screens show the noise each of them holds.",
    page: "/show",
    async run(r) {
      await startPipeline(r);
      await r.click("Picture");
      await r.recordUntil(pipelineDone, 40, 2);
    },
  },
  {
    id: "pipeline-clip",
    group: "pipeline",
    file: "pipeline-clip.gif",
    title: "Clip",
    caption: "Twelve frames of a short clip, two steps per frame, then the finished clip playing.",
    page: "/show",
    async run(r) {
      await startPipeline(r);
      await r.click("Clip");
      await r.skip(6.6);
      await r.recordUntil(pipelineDone, 40, 3);
    },
  },
  {
    id: "pipeline-own-text",
    group: "pipeline",
    file: "pipeline-own-text.gif",
    title: "Your text",
    caption: "Text typed by the reader; a toy model writes it backwards, one character per step, on random pieces.",
    page: "/show",
    viewport: { width: 1440, height: 1200 },
    async run(r) {
      await startPipeline(r, [".show-pick", STAGE]);
      await r.click("Your text");
      await r.frame([".show-pick", STAGE]);
      await r.record(0.5);
      await r.type("Type something", "my private note");
      await r.recordUntil(pipelineDone, 20, 2);
    },
  },
  {
    id: "pipeline-own-drawing",
    group: "pipeline",
    file: "pipeline-own-drawing.gif",
    title: "Your drawing",
    caption: "A drawing made on the pad is mirrored and embossed by a toy model; then the ready-made drawings.",
    page: "/show",
    viewport: { width: 1440, height: 1200 },
    async run(r) {
      await startPipeline(r, [".show-pick", STAGE]);
      await r.click("Your drawing");
      await r.frame([".show-pick", STAGE]);
      await r.click("Clear");
      await r.record(0.6);
      await r.drag(".show-pad", [0.15, 0.25], [0.85, 0.75], { steps: 7, as: "Drawing pad" });
      await r.drag(".show-pad", [0.15, 0.75], [0.85, 0.25], { steps: 7 });
      await r.record(4.5);
      await r.click("Arrow");
      await r.record(3.2);
      await r.click("Smile");
      await r.record(3.2);
    },
  },
  {
    id: "pipeline-chapter",
    group: "pipeline",
    file: "pipeline-chapter.gif",
    title: "Chapter size",
    caption: "The same pipeline as it appears inside chapters 5 and 7 (the first seconds of a run).",
    page: "/05-system-architecture",
    stage: "pipeline-chapter",
    async run(r) {
      await startPipeline(r);
      await r.record(13);
    },
  },
  {
    id: "pipeline-choices",
    group: "pipeline",
    file: "pipeline-choices.gif",
    title: "Prepared prompts and pictures",
    caption: "The six prepared questions and the two prepared pictures, chosen one after another.",
    page: "/show",
    viewport: { width: 1440, height: 1200 },
    async run(r) {
      await startPipeline(r, [".show-pick", STAGE]);
      for (const name of await r.names("Prepared prompts")) {
        await r.click(name);
        await r.record(1.6);
      }
      await r.click("Picture");
      for (const name of await r.names("Prepared pictures")) {
        await r.click(name);
        await r.record(2);
      }
    },
  },
  {
    id: "pipeline-playback",
    group: "pipeline",
    file: "pipeline-playback.gif",
    title: "Pause, step, speed, restart",
    caption: "The playback controls; then the panel that lists what is real and what is prepared is closed and opened (only its first lines are in the picture).",
    page: "/show",
    viewport: { width: 1440, height: 1500 },
    async run(r) {
      await r.ready(".theater");
      // The panel is long. For the recording only its first lines are kept, so the stage stays a readable size.
      await r.page.addStyleTag({ content: ".show-facts { max-height: 210px; overflow: hidden; }" });
      await r.settle(300);
      await r.frame([STAGE, ".show-details"]);
      await r.hold();
      await r.click("Restart");
      await r.unmark();
      await r.record(1.4);
      await r.click("Pause");
      await r.record(0.6);
      for (let index = 0; index < 3; index += 1) { await r.click("Next step"); await r.record(0.7); }
      await r.click("Play");
      await r.click("Speed: 1×");
      await r.record(2.4);
      await r.click("Speed: 2×");
      await r.record(1.2);
      await r.click("Restart");
      await r.record(1.4);
      const facts = r.page.locator(".show-details summary");
      await r.mark(facts);
      await facts.click();
      r.used("What is real here");
      await r.record(1.2);
      await r.mark(facts);
      await facts.click();
      await r.record(2.2);
    },
  },
  {
    id: "pipeline-full-screen",
    group: "pipeline",
    file: "pipeline-full-screen.gif",
    title: "Full screen",
    caption: "The demonstration filling the window.",
    page: "/show",
    viewport: { width: 1280, height: 800 },
    scale: 0.6,
    async run(r) {
      await r.ready(".theater");
      await r.frameWindow();
      await r.hold();
      await r.click("Restart");
      await r.record(1.2);
      await r.click("Full screen");
      await r.settle(600);
      if (!(await r.page.evaluate(() => Boolean(document.fullscreenElement)))) r.notes.push("this browser did not enter real full screen; the page's own full-window layout is shown");
      await r.record(5);
      await r.click("Exit full screen");
      await r.settle(400);
      await r.record(1.4);
    },
  },
  ...EYES.map((eye) => ({
    id: `eyes-${eye.id}`,
    group: "eyes",
    file: `eyes-${eye.id}.gif`,
    title: eye.label,
    caption: eye.says,
    page: "/show",
    async run(r) {
      await streamingPipeline(r, [".show-eyes", STAGE]);
      await r.record(1);
      await r.click(eye.label);
      await r.record(6);
    },
  })),
  ...ATTACKS.map((attack) => ({
    id: `attack-${attack.id}`,
    group: "attacks",
    file: `attack-${attack.id}.gif`,
    title: attack.label,
    caption: attack.says,
    page: "/show",
    viewport: { width: 1440, height: 1100 },
    async run(r) {
      await streamingPipeline(r, PIPELINE_CONTROLS);
      await r.record(0.8);
      await r.click(attack.label);
      await r.recordUntil(() => Number(document.querySelector('[data-testid="rejected"]')?.textContent ?? 0) > 0, 8, 4.2);
      if ((await r.page.evaluate(rejections)) === 0) throw new Error("the attack was not counted as rejected");
    },
  })),
  ...labRecordings,
  {
    id: "strength-show",
    group: "strength",
    file: "strength-show.gif",
    title: "How hard is this to undo?",
    caption: "Under the demonstration: readable bytes beside sealed ones, wrong keys the page really tries against them (none opens), what one changed bit does, and where each of these facts stops.",
    page: "/show",
    viewport: { width: 1440, height: 1300 },
    async run(r) {
      await r.ready(".theater");
      await r.hide(".show-details");
      await r.frame(".show-strength");
      await r.hold();
      await r.record(7);
    },
  },
  {
    id: "tour",
    group: "tour",
    file: "tour.gif",
    title: "Seven layers, one at a time",
    caption: "Split, Separate, Enclave, Check, Seal and route, Wipe, Prove: each with its guard post lit around the data.",
    page: "/protection",
    stage: "tour",
    async run(r) {
      await r.ready(".stage");
      await r.frame([".tour-layers", ".stage .scene3d-frame"]);
      await r.hold();
      for (const name of (await r.names("Layers of protection")).filter((text) => !/^(Play the tour|Pause)$/.test(text))) {
        const label = name.replace(/^\d+\s*/, "");
        await r.click(new RegExp(`^(\\d+\\s*)?${label}$`), { as: `Layer: ${label}` });
        await r.record(2.5);
      }
      await r.click("Play the tour");
      await r.record(1.2);
    },
  },
  ...DIAGRAMS.map((diagram) => ({
    id: `diagram-${diagram.id}`,
    group: "diagrams",
    file: `diagram-${diagram.id}.gif`,
    title: diagram.title,
    caption: diagram.says,
    page: diagram.page,
    viewport: { width: 1440, height: 1700 },
    colors: 64,
    fuzz: 8,
    diagram: diagram.id,
    async run(r) {
      const box = "details.schematic";
      await r.page.locator(`${box} > summary`).click();
      r.used("Schematic drawing, step by step");
      await r.settle(300);
      // Open one part first so that the recorded area is tall enough for it, then close it again.
      const parts = await r.names("Look inside a part", box);
      await r.page.locator(box).getByRole("button", { name: parts[0], exact: true }).first().click();
      await r.settle(200);
      await r.frame(box);
      await r.page.locator(box).getByRole("button", { name: parts[0], exact: true }).first().click();
      await r.hold();
      await r.click("Play walk-through", { scope: box });
      await r.record(5.6);
      await r.click("Pause", { scope: box });
      for (let index = 0; index < 4; index += 1) { await r.click("Next", { scope: box }); await r.record(1.3); }
      await r.click("Back", { scope: box });
      await r.record(0.9);
      await r.click("Show all", { scope: box });
      await r.record(1.4);
      for (const part of parts.slice(0, 2)) {
        await r.click(part, { scope: box, as: `Look inside: ${part}` });
        await r.record(1.6);
      }
    },
  })),
  {
    id: "setting-question",
    group: "settings",
    file: "setting-question.gif",
    title: "Your question",
    caption: "Six prepared questions. The choice is shared by the chapters, the laboratories, the diagrams and the tour.",
    page: "/05-system-architecture",
    async run(r) {
      await r.ready(".theater");
      await r.frame([".show-lead", STAGE]);
      await r.hold();
      const options = await r.page.locator(".show-lead select option").evaluateAll((list) => list.map((entry) => ({ value: entry.value, text: entry.textContent.trim() })));
      for (const entry of [...options.slice(1), options[0]]) {
        await r.select("Your question", entry.value, { scope: ".show-lead", as: `Your question: ${entry.text}` });
        await r.record(1.7);
      }
    },
  },
  {
    id: "setting-picture",
    group: "settings",
    file: "setting-picture.gif",
    title: "Your picture request",
    caption: "Two prepared picture requests, used by the picture laboratory.",
    page: "/laboratory#diffusion",
    async run(r) {
      await openLab(r, "diffusion");
      await r.frame("#diffusion .scene3d-frame");
      await r.hold();
      const options = await r.page.locator("#diffusion .stage select option").evaluateAll((list) => list.map((entry) => ({ value: entry.value, text: entry.textContent.trim() })));
      for (const entry of [...options.slice(1), options[0]]) {
        await r.select("Your picture request", entry.value, { scope: "#diffusion .stage", as: `Your picture request: ${entry.text}` });
        await r.record(3);
      }
    },
  },
  {
    id: "setting-words",
    group: "settings",
    file: "setting-words.gif",
    title: "Plain or technical words",
    caption: "Everyday wording, or the terms the chapters use.",
    page: "/show",
    viewport: { width: 1440, height: 1100 },
    async run(r) {
      await streamingPipeline(r, [STAGE, ".show-controls"]);
      await r.record(1.2);
      await r.click("Words: plain");
      await r.record(3);
      await r.click("Words: technical");
      await r.record(1.8);
    },
  },
  {
    id: "setting-effects",
    group: "settings",
    file: "setting-effects.gif",
    title: "Glow effects: on, off, automatic",
    caption: "Glow and shadows can be switched off; “auto” switches them off by itself on slow graphics hardware, as it does on the machine that made these recordings.",
    page: "/show",
    async run(r) {
      await streamingPipeline(r, ".theater .stage");
      await r.record(1.2);
      await r.click("Effects: on");
      await r.record(2.4);
      await r.click("Effects: off");
      await r.record(2.4);
      await r.click("Effects: auto");
      await r.record(2);
    },
  },
  {
    id: "setting-motion",
    group: "settings",
    file: "setting-motion.gif",
    title: "Pause motion",
    caption: "The 3D view stands still on request (and always does for readers who ask their system for reduced motion).",
    page: "/show",
    async run(r) {
      await streamingPipeline(r, ".theater .stage");
      await r.record(1.6);
      await r.click("Pause motion");
      await r.record(2.6);
      await r.click("Play motion");
      await r.record(2);
    },
  },
  {
    id: "setting-view",
    group: "settings",
    file: "setting-view.gif",
    title: "Turn, zoom and reset the 3D view",
    caption: "Every 3D view can be dragged round, zoomed and reset.",
    page: "/show",
    async run(r) {
      await streamingPipeline(r, ".theater .stage");
      await r.record(0.8);
      await r.drag(`${STAGE} canvas`, [0.45, 0.55], [0.7, 0.45], { steps: 10, as: "Drag to turn" });
      await r.record(1);
      for (const name of ["Zoom in", "Zoom in", "Zoom out", "Zoom out"]) { await r.click(name, { scope: ".theater" }); await r.record(0.7); }
      await r.click("Reset view", { scope: ".theater" });
      await r.record(1.6);
    },
  },
  {
    id: "setting-theme",
    group: "settings",
    file: "setting-theme.gif",
    title: "Light and dark theme",
    caption: "The pages switch between a light and a dark theme; the stages are dark in both.",
    page: "/laboratory#share-grid",
    viewport: { width: 1280, height: 860 },
    scale: 0.6,
    theme: "light",
    async run(r) {
      await openLab(r, "share-grid");
      await r.page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
      await r.frameWindow();
      await r.hold();
      await r.record(1.4);
      await r.click("Switch to dark theme", { as: "Dark theme" });
      await r.record(2.6);
      await r.click("Switch to light theme", { as: "Light theme" });
      await r.record(1.8);
    },
  },
  {
    id: "setting-lab-fullscreen",
    group: "settings",
    file: "setting-lab-fullscreen.gif",
    title: "A laboratory filling the window",
    caption: "Every laboratory has a Fullscreen button.",
    page: "/laboratory#share-grid",
    viewport: { width: 1280, height: 860 },
    scale: 0.6,
    async run(r) {
      await openLab(r, "share-grid");
      await r.page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
      await r.frameWindow();
      await r.hold();
      await r.record(1);
      await r.click("Fullscreen", { scope: "#share-grid" });
      await r.record(3);
      await r.click("Exit fullscreen", { scope: "#share-grid" });
      await r.record(1.2);
    },
  },
];
