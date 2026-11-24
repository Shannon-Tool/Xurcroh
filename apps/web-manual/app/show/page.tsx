import type { Metadata } from "next";
import Link from "next/link";
import { LiveShow } from "@/components/LiveShow";

export const metadata: Metadata = {
  title: "Live demonstration",
  description: "A prepared answer, picture and clip produced step by step on additive shares in your browser, with a 3D view of each step.",
};

export default function ShowPage() {
  return (
    <article className="article article-wide">
      <LiveShow autoPlay heading="h1" factsOpen />
      <p className="small">
        Each mechanism has its own laboratory with exact numbers: see the <Link href="/laboratory">mounted laboratory</Link>,
        or read the <Link href="/05-system-architecture">system architecture</Link>.
      </p>
    </article>
  );
}
