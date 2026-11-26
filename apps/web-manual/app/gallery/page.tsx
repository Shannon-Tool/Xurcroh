import type { Metadata } from "next";
import { Gallery } from "@/components/Gallery";

export const metadata: Metadata = {
  title: "All animations",
  description: "Every animated stage of the manual on one page, for review.",
};

export default function GalleryPage() {
  return (
    <article className="article article-wide">
      <header className="hero hero-compact">
        <span className="evidence">Review page · production-validated engine · interactive teaching simulations</span>
        <h1>All animations</h1>
        <p>
          Every animated stage in the manual, in one place, so that they can be compared and polished. Each one runs by itself while it is on
          screen. Under each title is the file that defines it and the page where readers meet it. The questions and answers are prepared, the
          relays and enclaves are stand-ins, and the arithmetic and encryption are computed in your browser.
        </p>
      </header>
      <Gallery />
    </article>
  );
}
