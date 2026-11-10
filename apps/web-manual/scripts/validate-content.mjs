import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import katex from "katex";
import mermaid from "mermaid";

const root = process.cwd();
const expected = [
  "01-introduction", "02-research-foundations", "03-threat-model", "04-mathematics",
  "05-system-architecture", "06-model-sharding", "07-inference-runtime", "08-secure-transport",
  "09-memory-security", "10-dense-transformers", "11-moe-architecture", "12-vision-language",
  "13-image-diffusion", "14-video-generation", "15-interactive-laboratory", "16-performance",
  "17-security-evaluation", "18-deployment", "19-developer-reference", "20-troubleshooting",
  "21-research-references", "22-experimental-reports",
];

for (const section of expected) {
  const path = join(root, "content", `${section}.mdx`);
  const text = await readFile(path, "utf8");
  if (text.length < 400) throw new Error(`${section} lacks substantive content`);
  for (const match of text.matchAll(/<Equation\s+tex="([^"]+)"\s*\/>/g)) {
    katex.renderToString(match[1], { displayMode: true, throwOnError: true, strict: "error" });
  }
  for (const match of text.matchAll(/href="(\/[^"]+)"/g)) {
    const target = match[1].split("#")[0];
    if (target !== "/" && !expected.includes(target.slice(1))) throw new Error(`${section} has broken link ${target}`);
  }
}

await mermaid.parse("flowchart LR\nClient --> Controller --> Worker");
console.log(`Validated ${expected.length} sections, equations, links, and Mermaid syntax`);
