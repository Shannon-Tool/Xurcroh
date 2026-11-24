import Link from "next/link";
import { ArchitectureExplorer } from "@/components/ArchitectureExplorer";
import { sections } from "@/lib/content";

export default function Home() {
  return (
    <article className="article">
      <header className="hero">
        <span className="evidence">Research manual · pre-GPU implementation</span>
        <h1>Distributed inference, examined end to end.</h1>
        <p>
          A rigorous technical manual for Xurcroh’s algebraic transformations, encrypted model custody,
          distributed runtime, secure transport, multimodal execution, memory lifecycle, and open
          security limitations. Every interactive example computes real state from defined inputs.
        </p>
        <p className="hero-actions">
          <Link href="/laboratory" className="button-link primary">Open the laboratory</Link>
          <Link href="/05-system-architecture" className="button-link">Read the architecture</Link>
        </p>
      </header>
      <section className="overview" aria-labelledby="overview-title">
        <h2 id="overview-title">The design in one picture</h2>
        <p className="small">
          One protected computation, step by step: your device splits its data into shares, four independent workers each compute on
          one share, and your device verifies and combines the results. This is an illustration of the design, designed for defence
          in depth with claims under evaluation. It is not a live system and shows no measurements.
        </p>
        <ArchitectureExplorer autoPlay variant="hero" />
      </section>
      <section className="home-grid" aria-label="Manual sections">
        {sections.map((section, index) => (
          <Link href={`/${section.slug}`} className="card" key={section.slug}>
            <span className="evidence">{String(index + 1).padStart(2, "0")} · {section.evidence}</span>
            <h2>{section.title}</h2>
            <p>{section.summary}</p>
          </Link>
        ))}
      </section>
    </article>
  );
}
