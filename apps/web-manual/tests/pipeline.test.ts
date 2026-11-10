import { describe, expect, it } from "vitest";
import { simulatePipeline } from "@/lib/simulation/pipeline";

describe("distributed pipeline simulation", () => {
  it("models computation, failure recovery, decoding, and cleanup", () => {
    const events = simulatePipeline({ nodes: 3, latencyMs: 2, packetLoss: 0, requests: 2, failNode: 1 });
    expect(events.some((event) => event.state === "recovered")).toBe(true);
    expect(events.filter((event) => event.state === "scrubbed")).toHaveLength(2);
    expect(events.filter((event) => event.state === "decoded")).toHaveLength(2);
  });

  it("adds retransmission events for packet loss", () => {
    const events = simulatePipeline({ nodes: 4, latencyMs: 1, packetLoss: 1, requests: 1, failNode: null });
    expect(events.filter((event) => event.state === "resent")).toHaveLength(4);
  });

  it("uses actual configurable latency", () => {
    const fast = simulatePipeline({ nodes: 2, latencyMs: 1, packetLoss: 0, requests: 1, failNode: null });
    const slow = simulatePipeline({ nodes: 2, latencyMs: 5, packetLoss: 0, requests: 1, failNode: null });
    expect(slow.at(-1)!.timeMs).toBeGreaterThan(fast.at(-1)!.timeMs);
  });
});
