#!/usr/bin/env node
// Writes the "See it run" section of the repository README from the list of recordings.
//   node scripts/render-readme.mjs
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { placeSection, renderSection } from "./lib/readme.mjs";
import { GROUPS, RECORDINGS } from "./recordings.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const index = JSON.parse(await readFile(path.join(root, "docs/media/animations/index.json"), "utf8"));
const missing = RECORDINGS.filter((recording) => !index[recording.id]);
if (missing.length > 0) throw new Error(`Not recorded yet: ${missing.map((recording) => recording.id).join(", ")}`);
const file = path.join(root, "README.md");
await writeFile(file, placeSection(await readFile(file, "utf8"), renderSection(RECORDINGS, GROUPS, index)));
console.log(`README.md: ${RECORDINGS.length} recordings in ${GROUPS.length} groups`);
