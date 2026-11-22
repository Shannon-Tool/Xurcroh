import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fireEvent, render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it } from "vitest";
import { TraceVerifier } from "@/components/TraceVerifier";

const fixture = JSON.parse(
  readFileSync(resolve("tests/fixtures/verify.json"), "utf8"),
);
const paste = (label: RegExp, value: string) =>
  fireEvent.change(screen.getByLabelText(label), { target: { value } });

describe("trace verification page", () => {
  it("verifies a pasted trace and says what an unpinned signer means", async () => {
    const { container } = render(<TraceVerifier />);
    expect(screen.getByRole("button", { name: "Verify" })).toBeDisabled();
    paste(/^Trace/, fixture.trace_text);
    fireEvent.click(screen.getByRole("button", { name: "Verify" }));
    expect(screen.getByRole("status")).toHaveTextContent("Verified.");
    expect(screen.getByRole("status")).toHaveTextContent(/re-signed by someone else/);
    expect(screen.getByRole("status")).toHaveTextContent(/2 worker checkpoint signatures verify/);
    expect((await axe(container)).violations).toHaveLength(0);
  });

  it("names the failing part and shows no success for a tampered trace", () => {
    render(<TraceVerifier />);
    paste(/^Trace/, fixture.trace_text.replace('"local"', '"hosted"'));
    fireEvent.click(screen.getByRole("button", { name: "Verify" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/signature is missing or does not verify/);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("warns that a hosted trace is the operator's statement", () => {
    render(<TraceVerifier />);
    paste(/^Trace/, JSON.stringify(fixture.media_trace));
    fireEvent.click(screen.getByRole("button", { name: "Verify" }));
    expect(screen.getByRole("status")).toHaveTextContent(/gateway operator's signed statement/);
    expect(screen.getByRole("status")).toHaveTextContent(/No worker signed a checkpoint/);
  });
});
