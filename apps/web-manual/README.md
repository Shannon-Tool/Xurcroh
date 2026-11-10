# Xurcroh Interactive Research Manual

A production Next.js/TypeScript manual for the actual Xurcroh implementation. It contains 22 substantive sections, KaTeX derivations, Mermaid structural diagrams, and seven deterministic interactive laboratories. Simulation code is documentation-only and is never imported by production inference paths.

## Develop and validate

```bash
npm ci
npm run dev
npm run check
```

`check` runs strict TypeScript, ESLint, simulation and interaction/accessibility tests, the production build, and content/equation/link/Mermaid validation.

## Structure

- `content/`: MDX sections with evidence labels and citations.
- `lib/simulation/`: deterministic engines, independently tested.
- `components/labs/`: accessible interactive visualizations.
- `components/`: playback, search, themes, matrices, and lab shell.
- `tests/`: engine, interaction, playback, and accessibility tests.
- `scripts/validate-content.mjs`: section, math, link, and Mermaid checks.

All labs support play/pause, step forward/backward, reset, speed, parameter changes, state inspection, and reduced-motion preferences. Synthetic simulations are not benchmark results or model-quality outputs. Physical GPU/TEE claims remain explicitly hardware-pending.
