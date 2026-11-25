// Builds the "See it run" section of the repository README from the list of recordings and the
// facts the recorder wrote down about each file.

export const START = "<!-- animations:start -->";
export const END = "<!-- animations:end -->";
const MEDIA = "docs/media/animations";

/** A control's label without its current value: "Latency 3 ms" and "Latency 20 ms" are the same control. */
export function plainLabel(label) {
  return label.replace(/\s+/g, " ").replace(/[=:]?\s*[-0-9.,]+\s*(ms|%|π)?\s*$/u, "").trim();
}

const escape = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function cell(recording, facts) {
  const lines = [
    `<img src="${MEDIA}/${recording.file}" alt="${escape(`${recording.title}. ${recording.caption}`).replace(/"/g, "&quot;")}" width="100%">`,
    `<b>${escape(recording.title)}</b>`,
    escape(recording.caption),
  ];
  if (recording.group === "lab-controls" && facts?.controls?.length) lines.push(`<sub>Controls used: ${escape(facts.controls.join(" · "))}</sub>`);
  if (facts?.notes?.length) lines.push(`<sub>Note: ${escape(facts.notes.join("; "))}.</sub>`);
  lines.push(`<sub>From <code>${escape(recording.page)}</code></sub>`);
  return `<td width="50%" valign="top">\n${lines.join("<br>\n")}\n</td>`;
}

/** The whole section, markers included. `index` is the recorder's `index.json`. */
export function renderSection(recordings, groups, index) {
  const lead = recordings.find((recording) => recording.lead);
  const megabytes = recordings.reduce((sum, recording) => sum + (index[recording.id]?.bytes ?? 0), 0) / 1e6;
  const out = [
    START,
    "## See it run",
    "",
    "These are recordings of the interactive manual in `apps/web-manual`. They show a simulation that runs in the reader's browser, not the production engine: the questions and answers are prepared, the models are toys, the three relays stand in for a Tor circuit, and the enclaves are simulated. What is computed for real in the page is the share arithmetic, the three layers of AES-GCM, the signature check on each worker's report and the check on each result. The pace is set for viewing and measures nothing.",
    "",
    `There are ${recordings.length} recordings (${megabytes.toFixed(0)} MB together), made frame by frame with \`npm run record\` in \`apps/web-manual\`. In the manual itself every one of these views is interactive.`,
    "",
  ];
  if (lead) {
    out.push(`<p align="center"><img src="${MEDIA}/${lead.file}" alt="${escape(`${lead.title}. ${lead.caption}`)}" width="100%"></p>`, "", `**${lead.title}.** ${lead.caption} From \`${lead.page}\`.`, "");
  }
  for (const group of groups) {
    const members = recordings.filter((recording) => recording.group === group.id && !recording.lead);
    if (members.length === 0) continue;
    out.push(`### ${group.title}`, "", group.lead, "", "<table>");
    for (let at = 0; at < members.length; at += 2) {
      const pair = members.slice(at, at + 2);
      out.push("<tr>", ...pair.map((recording) => cell(recording, index[recording.id])), ...(pair.length === 1 ? ['<td width="50%"></td>'] : []), "</tr>");
    }
    out.push("</table>", "");
  }
  out.push(END);
  return out.join("\n");
}

/** Puts the section into a README: in place of the old one, or after the line that starts with `after`. */
export function placeSection(readme, section, after = "> **Security status:**") {
  const from = readme.indexOf(START);
  const to = readme.indexOf(END);
  if (from >= 0 && to > from) return `${readme.slice(0, from)}${section}${readme.slice(to + END.length)}`;
  const lines = readme.split("\n");
  const anchor = lines.findIndex((line) => line.startsWith(after));
  if (anchor < 0) throw new Error(`The README has no line starting with “${after}”`);
  lines.splice(anchor + 1, 0, "", section);
  return lines.join("\n");
}
