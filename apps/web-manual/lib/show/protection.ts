/** The protection told as layers. Each one names the threat, the mechanism, what it relies on and where it stops. */
export interface ProtectionLayer {
  id: string;
  name: string;
  /** What the layer achieves, in a few words. */
  claim: string;
  threat: string;
  mechanism: string;
  relies: string;
  limit: string;
  /** Chapter that documents it; its evidence tag is shown with the layer. */
  chapter: string;
  /** Laboratory whose live strip demonstrates it. */
  lab?: "share-grid" | "placement" | "freivalds" | "onion" | "memory";
}

export const PROTECTION_LAYERS: ProtectionLayer[] = [
  {
    id: "split", name: "Split", claim: "No worker gets your data",
    threat: "A machine that runs the model for you could read your prompt.",
    mechanism: "Your device cuts every number into two random pieces and never sends both pieces to the same worker.",
    relies: "Fresh randomness for every split, and workers with different pieces not pooling them.",
    limit: "Two workers holding matching pieces can rebuild your data together.",
    chapter: "05-system-architecture", lab: "share-grid",
  },
  {
    id: "separate", name: "Separate", claim: "No operator holds a matching pair",
    threat: "One company running many workers could quietly collect both pieces.",
    mechanism: "The planner places the pieces of each operation with different providers, and moves them around again and again.",
    relies: "The providers being independent operators.",
    limit: "Providers that cooperate defeat it. Placement makes that harder; it cannot rule it out.",
    chapter: "18-deployment", lab: "placement",
  },
  {
    id: "enclave", name: "Enclave", claim: "The machine's operator cannot look in",
    threat: "The company that runs a worker's machine could read its memory while it computes.",
    mechanism: "Every worker must run inside a hardware enclave (TEE) and show a signed report of the software it runs before it is sent anything.",
    relies: "The processor and its maker. An enclave is one more layer, not proof.",
    limit: "Enclaves have been broken by physical attacks, and attestation of workers has not yet been validated on real hardware.",
    chapter: "18-deployment",
  },
  {
    id: "check", name: "Check", claim: "Wrong answers are caught",
    threat: "A worker could send back a wrong or doctored answer.",
    mechanism: "Your device tests every answer against a random challenge before it uses it. A failed test throws the step away.",
    relies: "The worker not being able to predict the challenge, and exact arithmetic.",
    limit: "It shows an answer is correct. It does not show the worker kept anything private.",
    chapter: "17-security-evaluation", lab: "freivalds",
  },
  {
    id: "seal", name: "Seal and route", claim: "The network sees only sealed bytes",
    threat: "Someone watching the network, or a relay on the way, could read or trace what you send.",
    mechanism: "Everything travels in layers of encryption through Tor. A relay can open only its own layer and learns only the next stop.",
    relies: "Relays not sharing what they see with each other, and standard encryption. The Tor carrier has run on the real network.",
    limit: "It does not hide timing or how much is sent, and relays that work together can link a route.",
    chapter: "08-secure-transport", lab: "onion",
  },
  {
    id: "wipe", name: "Wipe", claim: "Nothing is left in memory",
    threat: "The next job on the same graphics card could read what your request left behind.",
    mechanism: "Memory is wiped, and the wipe is confirmed finished, before the memory is handed to anyone else.",
    relies: "The wipe completing on every pool that held your data.",
    limit: "A software wipe does not prove that every physical register inside the chip is cleared.",
    chapter: "09-memory-security", lab: "memory",
  },
  {
    id: "prove", name: "Prove", claim: "You can check it yourself",
    threat: "Otherwise you would have to take the operator’s word for what ran.",
    mechanism: "Every request produces a signed trace: what was asked for, which model build ran, how the links were protected and what each worker signed for. You can verify one in your browser.",
    relies: "The signing keys being kept safe, and someone actually checking the trace.",
    limit: "A trace shows what was signed, not what a compromised machine did besides. Independent audit is still to come.",
    chapter: "17-security-evaluation",
  },
];

export const PROTECTION_LIMITS = [
  "Workers that pool matching pieces can rebuild your data. Everything rests on independent operators not doing that.",
  "Timing, traffic volume and hardware side channels are not hidden by padding or encryption.",
  "Hardware enclaves are treated as one more layer, not as proof.",
  "The claims are still under evaluation: independent attestation, reproduction by a second party, an external audit and red-team review have not been completed.",
];
