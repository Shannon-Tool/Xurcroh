import type { Metadata } from "next";
import Link from "next/link";
import { ProtectionTour } from "@/components/ProtectionTour";

export const metadata: Metadata = {
  title: "How your data is protected",
  description: "Seven layers of protection, each shown working on your own input, with what it relies on and where it stops.",
};

export default function ProtectionPage() {
  return (
    <article className="article article-wide">
      <header className="hero hero-compact">
        <span className="evidence">Designed for defence in depth · claims under evaluation</span>
        <h1>How your data is protected</h1>
        <p>
          Seven layers stand between your data and anyone else. For each one this page names the threat, shows the mechanism working on
          input you type, and says plainly what it relies on and where it stops. Nothing here is a promise: it is what the design does
          and how far that has been checked.
        </p>
      </header>
      <h2>The layers</h2>
      <ProtectionTour autoPlay />
      <p className="small">
        Watch all of it work together in the <Link href="/show">live demonstration</Link>, or go through every mechanism with exact numbers in the{" "}
        <Link href="/laboratory">laboratory</Link>.
      </p>
    </article>
  );
}
