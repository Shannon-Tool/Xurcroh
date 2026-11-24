import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { sections } from "@/lib/content";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Search } from "@/components/Search";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Xurcroh Research Manual", template: "%s · Xurcroh" },
  description: "Interactive scientific and engineering manual for zero-trust distributed inference.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <header className="site-header">
          <Link href="/" className="brand">XURCROH / RESEARCH MANUAL</Link>
          <div className="header-actions">
            <ThemeToggle />
          </div>
        </header>
        <div className="layout">
          <aside className="sidebar" aria-label="Manual navigation">
            <Search />
            <nav>
              <Link href="/show" className="nav-link"><span className="nav-number">▶</span><span>Live demonstration</span></Link>
              <Link href="/laboratory" className="nav-link"><span className="nav-number">↳</span><span>Mounted laboratory</span></Link>
              <Link href="/verify" className="nav-link"><span className="nav-number">✓</span><span>Verify a task trace</span></Link>
              {sections.map((section, index) => (
                <Link key={section.slug} href={`/${section.slug}`} className="nav-link">
                  <span className="nav-number">{String(index + 1).padStart(2, "0")}</span>
                  <span>{section.title}</span>
                </Link>
              ))}
            </nav>
          </aside>
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
