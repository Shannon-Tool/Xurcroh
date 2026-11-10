import katex from "katex";

export function Equation({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, {
    displayMode: true,
    throwOnError: true,
    strict: "error",
    output: "html",
  });
  return <div className="equation" dangerouslySetInnerHTML={{ __html: html }} />;
}
