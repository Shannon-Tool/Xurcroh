# Xurcroh Interactive Research Manual

A production Next.js/TypeScript manual for the actual Xurcroh implementation. It contains 22 substantive sections, KaTeX derivations, structural diagrams with step-by-step walk-throughs, and eleven deterministic interactive laboratories, each with an animated 3D view. Simulation code is documentation-only and is never imported by production inference paths.

## Develop and validate

```bash
npm ci
npm run dev
npm run check
```

`check` runs strict TypeScript, ESLint, simulation and interaction/accessibility tests, the production build, and content/equation/link validation.

## Structure

- `content/`: MDX sections with evidence labels and citations.
- `lib/simulation/`: deterministic engines, independently tested.
- `components/labs/`: accessible interactive visualizations.
- `components/`: playback, search, themes, matrices, and lab shell.
- `tests/`: engine, interaction, playback, and accessibility tests.
- `scripts/validate-content.mjs`: section, math, and link checks.

Labs with more than one step support play/pause, step forward/backward, reset and speed; every lab has parameter controls, an animated, rotatable 3D view of its current state, and a text view. The 3D views pause on request and stay still under reduced-motion preferences; glow and shadow effects switch off automatically on slow graphics hardware or by choice. Synthetic simulations are not benchmark results or model-quality outputs. Physical GPU/TEE claims remain explicitly hardware-pending.

The home page and `/show` carry a live demonstration: a prepared answer, picture and short clip are produced one step at a time while a 3D view plays the matching protocol steps. Each step is one matrix product that is computed on additive shares, checked with a Freivalds test and recombined in the browser (`lib/simulation/show.ts`); the worker tiles show the share data of that step, and a tamper button makes the check reject a product. The models are toys (a lookup table for text, a hand-built linear decoder for pictures), no language or image model runs, and the pace measures nothing. The page says so next to the demonstration. Readers can also run it on their own text (a toy model writes it backwards) or their own drawing (mirrored and embossed), and switch “Whose eyes?” to see what one worker, two workers together, or a network observer would get, rebuilt from what that party holds.

Every animation is shown on one stage design (`components/StageBox.tsx`): a dark 16:9 box with the 3D scene in the middle, what goes in on the left, what comes out on the right, and along the bottom a caption and a ribbon showing the data in each form it takes. Stages play and loop by themselves while on screen, and stand still under reduced motion. `/gallery` puts every stage on one page for review.

The pipeline stage (`/show`, the home page, chapters 5 and 7) runs eight steps: split, attest, seal, route, compute, return, check, combine. The input is shuffled with a secret order (a disguise, not encryption), split into additive shares, wrapped in three layers of AES-GCM and passed through three relays that each remove one layer, and computed by four workers that each first present a signed report (ECDSA over the hash of their code and a fresh nonce). The relays stand in for a Tor circuit and the enclaves are simulated; both say so on the stage, and the page states that a TEE is one more layer, not proof, and that worker attestation has not yet been validated on hardware. Three attacks can be tried: changing a worker's result, swapping a worker's software, and altering a sealed packet; each is caught by the real check.

Every laboratory is told as one trip from a question to the model's answer. The reader picks one of six prepared questions (the choice is shared by all labs, the diagrams and the protection tour), and each lab shows three cards: the question going in, what the other party sees, and the model's full answer coming out, with the lab's own result under it. A bar shows which stop of the trip the lab covers, and the cards are followed by why that protects the reader, what it relies on, and where it stops. Labs with an attack offer it as a button; a blocked step withholds the answer. The first step of answering, scoring the question's word counts against the prepared answers, really runs on additive shares (`lib/show/journey.ts`), and the labs and diagrams quote its numbers. The questions and answers themselves are prepared: no language or image model runs. The chapter diagrams follow the same question: the real value at each step rides the arrows, any part can be opened to see what it holds, and the full answer is shown underneath. `/protection` tells the protection as seven layers, each with its threat, mechanism, assumption, limit, the chapter's evidence tag, and the matching lab's live cards. None of these pages claims the system is safe; they show what each party can see and where each mechanism stops.
