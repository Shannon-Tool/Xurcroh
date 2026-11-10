export function LabLink({ href, children }: { href: string; children: React.ReactNode }) {
  return <a className="card" href={href}>{children}</a>;
}
