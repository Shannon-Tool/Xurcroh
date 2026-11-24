"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Evidence } from "@/components/Evidence";
import { LabEmbedContext } from "@/components/LabShell";
import { StageBox } from "@/components/StageBox";
import { FreivaldsLab } from "@/components/labs/FreivaldsLab";
import { MemoryLab } from "@/components/labs/MemoryLab";
import { OnionLab } from "@/components/labs/OnionLab";
import { PlacementLab } from "@/components/labs/PlacementLab";
import { ShareGridLab } from "@/components/labs/ShareGridLab";
import { sections } from "@/lib/content";
import { protectionScene } from "@/lib/scene/protection-scene";
import { PROTECTION_LAYERS, PROTECTION_LIMITS, type ProtectionLayer } from "@/lib/show/protection";

const LABS: Record<NonNullable<ProtectionLayer["lab"]>, () => React.ReactNode> = {
  "share-grid": () => <ShareGridLab />,
  placement: () => <PlacementLab />,
  freivalds: () => <FreivaldsLab />,
  onion: () => <OnionLab />,
  memory: () => <MemoryLab />,
};

const EVIDENCE_WORDS: Record<string, string> = {
  derived: "derived from the design, not measured",
  local: "checked by automated tests on a development machine",
  limitation: "documented together with its limits",
  hardware: "checked on hardware",
  published: "from published work",
  estimate: "an estimate",
  hypothesis: "a hypothesis",
};

const LAYER_MS = 9000;

/**
 * The protection as one story: six layers around your data. Each layer names the threat it answers, shows the
 * mechanism working on the reader's own input, and says what it relies on and where it stops.
 */
export function ProtectionTour({ autoPlay = false }: { autoPlay?: boolean }) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const layer = PROTECTION_LAYERS[index]!;
  const chapter = sections.find((section) => section.slug === layer.chapter);

  useEffect(() => {
    const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (autoPlay && !reduced) setPlaying(true);
  }, [autoPlay]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setIndex((current) => (current + 1) % PROTECTION_LAYERS.length), LAYER_MS);
    return () => window.clearInterval(timer);
  }, [playing]);

  const scene = useMemo(() => protectionScene({ layer: index }), [index]);
  const choose = (next: number) => {
    setPlaying(false);
    setIndex(next);
  };

  return (
    <div className="tour">
      <div className="tour-layers playback" role="group" aria-label="Layers of protection">
        <button className="lab-watch" onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play the tour"}</button>
        {PROTECTION_LAYERS.map((entry, position) => (
          <button key={entry.id} className={position === index ? "tour-layer current" : "tour-layer"} aria-pressed={position === index} onClick={() => choose(position)}>
            <span aria-hidden="true">{position + 1}</span> {entry.name}
          </button>
        ))}
      </div>

      <StageBox
        label="The layers of protection, one at a time"
        scene={scene}
        autoRotate={false}
        left={(
          <div className="stage-panel stage-in">
            <p className="stage-label">Layer {index + 1} of {PROTECTION_LAYERS.length}</p>
            <p><strong>{layer.name}.</strong> {layer.claim}.</p>
          </div>
        )}
        right={(
          <div className="stage-panel stage-seen">
            <p className="stage-label">What it stops</p>
            <p>{layer.threat}</p>
          </div>
        )}
        caption={layer.mechanism}
        why={`It relies on: ${layer.relies.charAt(0).toLowerCase()}${layer.relies.slice(1)}`}
        forms={PROTECTION_LAYERS.map((entry) => ({ id: entry.id, label: entry.name, value: entry.claim, tone: entry.id === layer.id ? "accent" as const : "plain" as const }))}
        current={[layer.id]}
        ribbonLabel="The layers of protection"
      />

      <section className="tour-panel" aria-live="polite" aria-labelledby={`tour-${layer.id}`}>
        <h3 id={`tour-${layer.id}`}>{index + 1}. {layer.name}: {layer.claim.toLowerCase()}</h3>
        <dl className="tour-facts">
          <div className="threat"><dt>The threat</dt><dd>{layer.threat}</dd></div>
          <div><dt>What is done about it</dt><dd>{layer.mechanism}</dd></div>
          <div><dt>It relies on</dt><dd>{layer.relies}</dd></div>
          <div className="limit"><dt>Where it stops</dt><dd>{layer.limit}</dd></div>
        </dl>
        {chapter ? (
          <p className="tour-evidence">
            <Evidence type={chapter.evidence}>Evidence: {chapter.evidence}</Evidence> {EVIDENCE_WORDS[chapter.evidence] ?? chapter.evidence}, as tagged in{" "}
            <Link href={`/${chapter.slug}`}>chapter {chapter.slug.slice(0, 2)}, {chapter.title}</Link>.
          </p>
        ) : null}
        {layer.lab ? (
          <>
            <p className="tour-try">See it work for the question you choose. This is the live part of the <Link href={`/laboratory#${layer.lab}`}>laboratory</Link>:</p>
            <LabEmbedContext.Provider value="cards">{LABS[layer.lab]()}</LabEmbedContext.Provider>
          </>
        ) : layer.id === "enclave" ? (
          <p className="tour-try">Watch it in the <Link href="/show">live demonstration</Link>: press “Swap a worker’s software” and the worker’s signed report is refused. The enclaves there are simulated; the signature check is real.</p>
        ) : (
          <p className="tour-try">Check a real trace yourself on the <Link href="/verify">trace verification page</Link>. It runs in your browser and sends nothing anywhere.</p>
        )}
      </section>

      <section className="tour-limits" aria-labelledby="tour-limits-title">
        <h3 id="tour-limits-title">What this does not protect against</h3>
        <ul>{PROTECTION_LIMITS.map((entry) => <li key={entry}>{entry}</li>)}</ul>
        <p>The full list is in the <Link href="/03-threat-model">threat model</Link>. The design is built for defence in depth, and its claims are under evaluation.</p>
      </section>
    </div>
  );
}
