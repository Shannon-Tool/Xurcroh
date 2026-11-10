export interface PipelineConfig {
  nodes: number;
  latencyMs: number;
  packetLoss: number;
  requests: number;
  failNode: number | null;
}

export interface PipelineEvent {
  timeMs: number;
  request: number;
  stage: string;
  state: "encrypted" | "computed" | "resent" | "recovered" | "scrubbed" | "decoded";
  stageIndex: number;
}

function random(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let value = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function simulatePipeline(config: PipelineConfig): PipelineEvent[] {
  if (config.nodes < 1 || config.nodes > 8 || config.requests < 1 || config.requests > 8) {
    throw new Error("pipeline supports 1-8 nodes and requests");
  }
  if (config.latencyMs < 0 || config.packetLoss < 0 || config.packetLoss > 1) {
    throw new Error("invalid latency or packet loss");
  }
  const rng = random(0x58555243);
  const events: PipelineEvent[] = [];
  for (let request = 0; request < config.requests; request += 1) {
    let time = request * config.latencyMs * 0.5;
    events.push({ timeMs: time, request, stage: "client", state: "encrypted", stageIndex: -1 });
    let failed = false;
    for (let node = 0; node < config.nodes; node += 1) {
      time += config.latencyMs;
      const lost = rng() < config.packetLoss;
      if (lost) {
        time += config.latencyMs;
        events.push({ timeMs: time, request, stage: `worker-${node}`, state: "resent", stageIndex: node });
      }
      if (config.failNode === node && !failed) {
        failed = true;
        time += config.latencyMs;
        events.push({ timeMs: time, request, stage: `worker-${node}`, state: "recovered", stageIndex: node });
      }
      events.push({ timeMs: time, request, stage: `worker-${node}`, state: "computed", stageIndex: node });
    }
    time += config.latencyMs;
    events.push({ timeMs: time, request, stage: "client", state: "decoded", stageIndex: -1 });
    events.push({ timeMs: time, request, stage: `worker-${config.nodes - 1}`, state: "scrubbed", stageIndex: config.nodes - 1 });
  }
  return events;
}
