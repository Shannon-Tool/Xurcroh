import Link from "next/link";
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
      </header>
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
