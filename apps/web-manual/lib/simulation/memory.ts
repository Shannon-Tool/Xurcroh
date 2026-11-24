export interface MemoryStep {
  operation: "allocate" | "execute" | "release" | "synchronize" | "zero" | "reuse";
  bytes: number[];
  note: string;
}

export function simulateMemory(synchronized: boolean, sensitive: number[] = [0xde, 0xad, 0xbe, 0xef]): MemoryStep[] {
  return [
    { operation: "allocate", bytes: sensitive, note: "Request A writes a canary" },
    { operation: "execute", bytes: sensitive, note: "KV cache and activations are consumed" },
    { operation: "release", bytes: sensitive, note: "Logical blocks return to the allocator" },
    { operation: "synchronize", bytes: sensitive, note: synchronized ? "Wipe is ordered before reuse" : "Wipe is only queued" },
    { operation: "zero", bytes: synchronized ? sensitive.map(() => 0) : sensitive, note: synchronized ? "Physical slots read as zero" : "Residual data remains readable" },
    { operation: "reuse", bytes: synchronized ? sensitive.map(() => 0) : sensitive, note: synchronized ? "Request B cannot recover Request A" : "Request B recovers stale bytes" },
  ];
}
