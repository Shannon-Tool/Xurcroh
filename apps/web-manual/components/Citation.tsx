export function Citation({ id, title, url }: { id: string; title: string; url: string }) {
  return (
    <a className="citation" href={url} target="_blank" rel="noreferrer">
      <span className="citation-id">[{id}]</span>{title}
    </a>
  );
}
