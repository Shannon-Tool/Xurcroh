export const PRIME = (1n << 32n) - 5n;

export function mod(value: bigint, modulus: bigint = PRIME): bigint {
  return ((value % modulus) + modulus) % modulus;
}

function mask(index: number, epoch: number): bigint {
  let value = BigInt(index + 1) * 0x9e3779b1n + BigInt(epoch + 1) * 0x85ebca6bn;
  value ^= value >> 16n;
  value *= 0x7feb352dn;
  value ^= value >> 15n;
  return mod(value);
}

function shares(values: number[], epoch: number): bigint[][] {
  const first = values.map((_, index) => mask(index, epoch));
  const second = values.map((value, index) => mod(BigInt(value) - first[index]!));
  return [first, second];
}

/** Fixed teaching challenge, one entry per output row. Production draws it at random for every check. */
const CHALLENGE = [11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n, 41n, 43n, 47n, 53n];

export interface GridRole {
  activationShare: number;
  weightShare: number;
  product: bigint[];
  valid: boolean;
}

export function executeShareGrid(weight: number[][], activation: number[], epoch: number, tamper?: string) {
  // Each row gets its own masks, so no two rows of a weight share look alike.
  const weightShares = weight.map((row, rowIndex) => shares(row, epoch + rowIndex * 1009));
  const activationShares = shares(activation, epoch + 17);
  const roles: GridRole[] = [];
  for (let a = 0; a < 2; a += 1) {
    for (let w = 0; w < 2; w += 1) {
      const product = weight.map((row, rowIndex) => {
        const exact = row.reduce((sum, _, column) =>
          sum + weightShares[rowIndex]![w]![column]! * activationShares[a]![column]!, 0n);
        return tamper === `${a}:${w}` && rowIndex === 0 ? exact + 1n : exact;
      });
      const challenge = CHALLENGE.slice(0, weight.length);
      const weightShare = weight.map((row, rowIndex) => weightShares[rowIndex]![w]!);
      const valid = freivaldsCheck(weightShare, activationShares[a]!, product, challenge);
      if (!valid) throw new Error(`role ${a}:${w} failed exact Freivalds verification`);
      roles.push({ activationShare: a, weightShare: w, product, valid });
    }
  }
  const combined = weight.map((_, index) => mod(roles.reduce((sum, role) => sum + role.product[index]!, 0n)));
  return { weightShares, activationShares, roles, combined };
}

export function freivaldsCheck(weight: Array<Array<number | bigint>>, activation: Array<number | bigint>, product: Array<number | bigint>, challenge: Array<number | bigint>): boolean {
  if (weight.length !== product.length || challenge.length !== weight.length) return false;
  const v = weight[0]!.map((_, column) =>
    mod(weight.reduce<bigint>((sum, row, index) => sum + BigInt(row[column]!) * BigInt(challenge[index]!), 0n), PRIME));
  const left = mod(activation.reduce<bigint>((sum, value, index) => sum + BigInt(value) * v[index]!, 0n), PRIME);
  const right = mod(product.reduce<bigint>((sum, value, index) => sum + BigInt(value) * BigInt(challenge[index]!), 0n), PRIME);
  return left === right;
}

export interface PlacementRole {
  operation: string;
  activationShare: number;
  weightShare: number;
  provider: string;
  country: string;
  cpuVendor: string;
}

export function placeOperations(epoch: number): PlacementRole[] {
  const operations = ["query", "key", "gate", "up"];
  const roles: PlacementRole[] = [];
  operations.forEach((operation, operationIndex) => {
    for (let activationShare = 0; activationShare < 2; activationShare += 1) {
      for (let weightShare = 0; weightShare < 2; weightShare += 1) {
        const localRole = activationShare * 2 + weightShare;
        const provider = (operationIndex * 4 + localRole + epoch * 4) % 16;
        roles.push({
          operation,
          activationShare,
          weightShare,
          provider: `provider-${provider}`,
          country: provider % 2 === 0 ? "de" : "us",
          cpuVendor: provider % 3 === 0 ? "intel" : "amd",
        });
      }
    }
  });
  return roles;
}
