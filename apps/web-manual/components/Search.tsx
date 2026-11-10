"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { glossary, sections } from "@/lib/content";

export function Search() {
  const [query, setQuery] = useState("");
  const results = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return [];
    return sections.filter((section) => `${section.title} ${section.summary}`.toLowerCase().includes(normalized));
  }, [query]);
  const terms = useMemo(() => glossary.filter((entry) => entry.term.toLowerCase().includes(query.trim().toLowerCase())), [query]);

  return (
    <section aria-label="Search manual">
      <label className="search-label" htmlFor="manual-search">Search manual</label>
      <input id="manual-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="conjugation, MoE, QUIC…" />
      {query ? <div className="search-results" aria-live="polite">
        {results.map((section) => <Link key={section.slug} href={`/${section.slug}`}>{section.title}</Link>)}
        {results.length === 0 ? <span className="small">No sections found</span> : null}
      </div> : null}
      {query && terms.length > 0 ? <dl className="glossary">
        {terms.map((entry) => <div key={entry.term}><dt>{entry.term}</dt><dd>{entry.definition}</dd></div>)}
      </dl> : null}
    </section>
  );
}
