export function Evidence({ type, children }: { type: string; children: React.ReactNode }) {
  return <span className={`evidence evidence-${type}`}>{children}</span>;
}
