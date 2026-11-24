import Link from "next/link";
import { LiveShow } from "@/components/LiveShow";
import { PROTECTION_LAYERS } from "@/lib/show/protection";
import { sections } from "@/lib/content";

export default function Home() {
  return (
    <article className="article article-wide">
      <header className="hero hero-compact">
        <span className="evidence">Research manual · pre-GPU implementation</span>
        <h1>Distributed inference, examined end to end.</h1>
        <p className="hero-actions">
          <Link href="/laboratory" className="button-link primary">Open the laboratory</Link>
          <Link href="/05-system-architecture" className="button-link">Read the architecture</Link>
          <Link href="/show" className="button-link">Demonstration on its own page</Link>
        </p>
      </header>
      <LiveShow autoPlay />
      <section className="overview" aria-labelledby="layers-title">
        <h2 id="layers-title">How your data is protected</h2>
        <p>Six layers, each one shown working on your own input, with what it relies on and where it stops.</p>
        <ol className="layer-row">
          {PROTECTION_LAYERS.map((layer) => (
            <li key={layer.id}><Link href="/protection"><strong>{layer.name}</strong><span>{layer.claim}</span></Link></li>
          ))}
        </ol>
        <p className="hero-actions"><Link href="/protection" className="button-link primary">Take the tour</Link></p>
      </section>
      <section className="overview" aria-labelledby="overview-title">
        <h2 id="overview-title">About this manual</h2>
        <p>
          A rigorous technical manual for Xurcroh’s algebraic transformations, encrypted model custody,
          distributed runtime, secure transport, multimodal execution, memory lifecycle, and open
          security limitations. Every interactive example computes real state from defined inputs.
        </p>
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
