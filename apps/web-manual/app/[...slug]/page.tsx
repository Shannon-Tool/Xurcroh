import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { sections } from "@/lib/content";
import { mdxComponents } from "@/lib/mdx";

export function generateStaticParams() {
  return sections.map((section) => ({ slug: [section.slug] }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string[] }> }): Promise<Metadata> {
  const { slug } = await params;
  const section = sections.find((candidate) => candidate.slug === slug[0]);
  if (!section) notFound();
  return { title: section.title, description: section.summary };
}

export default async function ManualPage({ params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params;
  const section = sections.find((candidate) => candidate.slug === slug[0]);
  if (!section) notFound();
  const source = await readFile(join(process.cwd(), "content", section.file), "utf8");
  return (
    <article className="article">
      <MDXRemote
        source={source}
        components={mdxComponents}
        options={{
          mdxOptions: {
            remarkPlugins: [remarkGfm, remarkMath],
            rehypePlugins: [rehypeKatex],
          },
        }}
      />
    </article>
  );
}
