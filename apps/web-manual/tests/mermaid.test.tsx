import { render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const renderChart = vi.fn(async (id: string, chart: string) => ({ svg: `<svg data-id="${id}" data-lines="${chart.split("\n").length}"></svg>` }));
vi.mock("mermaid", () => ({ default: { initialize: vi.fn(), render: (id: string, chart: string) => renderChart(id, chart) } }));

import { Mermaid } from "@/components/Mermaid";

describe("Mermaid", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    renderChart.mockClear();
  });

  it("renders on a page that is not a secure context, where crypto.randomUUID is missing", async () => {
    vi.stubGlobal("crypto", {});
    const { container } = render(<Mermaid chart="flowchart LR\n  A --> B" caption="c" />);
    await waitFor(() => expect(container.querySelector("svg")).not.toBeNull());
    const id = renderChart.mock.calls[0]?.[0] ?? "";
    expect(id).toMatch(/^mermaid-[A-Za-z0-9_-]+$/);
  });

  it("turns the escaped line breaks written in chapter source into real ones", async () => {
    const { container } = render(<Mermaid chart={"flowchart LR\\n  A --> B\\n  B --> C"} caption="c" />);
    await waitFor(() => expect(container.querySelector("svg")?.getAttribute("data-lines")).toBe("3"));
  });
});
