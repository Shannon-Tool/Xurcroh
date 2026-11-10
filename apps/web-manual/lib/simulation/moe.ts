export interface MoePlan {
  tokenIndices: number[][];
  weights: number[][];
  dummy: boolean[][];
  loads: number[];
}

export function routeMoe(
  hidden: number[][],
  router: number[][],
  topK: number,
  constantLoad: number,
): MoePlan {
  const experts = router.length;
  if (topK < 1 || topK > experts) throw new Error("invalid top-k");
  const selections = hidden.map((token) => {
    const logits = router.map((expert) => expert.reduce((sum, value, i) => sum + value * token[i]!, 0));
    const ranked = logits.map((logit, expert) => ({ logit, expert })).sort((a, b) => b.logit - a.logit);
    const top = ranked.slice(0, topK);
    const maximum = Math.max(...top.map((entry) => entry.logit));
    const exponents = top.map((entry) => Math.exp(entry.logit - maximum));
    const total = exponents.reduce((sum, value) => sum + value, 0);
    return top.map((entry, index) => ({ expert: entry.expert, weight: exponents[index]! / total }));
  });
  const tokenIndices = Array.from({ length: experts }, () => Array(constantLoad).fill(0));
  const weights = Array.from({ length: experts }, () => Array(constantLoad).fill(0));
  const dummy = Array.from({ length: experts }, () => Array(constantLoad).fill(true));
  for (let expert = 0; expert < experts; expert += 1) {
    const locations: { token: number; weight: number }[] = [];
    selections.forEach((selection, token) => {
      selection.forEach((entry) => {
        if (entry.expert === expert) locations.push({ token, weight: entry.weight });
      });
    });
    if (locations.length > constantLoad) throw new Error("constant-load capacity exceeded");
    locations.forEach((entry, index) => {
      tokenIndices[expert]![index] = entry.token;
      weights[expert]![index] = entry.weight;
      dummy[expert]![index] = false;
    });
  }
  return { tokenIndices, weights, dummy, loads: Array(experts).fill(constantLoad) };
}

export function combineMoe(plan: MoePlan, outputs: number[][][]): number[][] {
  const tokenCount = Math.max(...plan.tokenIndices.flat()) + 1;
  const result = Array.from({ length: tokenCount }, () => Array(outputs[0]![0]!.length).fill(0));
  outputs.forEach((expertOutput, expert) => {
    expertOutput.forEach((output, slot) => {
      if (plan.dummy[expert]![slot]) return;
      const token = plan.tokenIndices[expert]![slot]!;
      output.forEach((value, channel) => {
        result[token]![channel] += value * plan.weights[expert]![slot]!;
      });
    });
  });
  return result;
}
