export type QuantizedEncoding = "gguf-q4_0" | "exl2-mixed" | "exl3-3inst";

export type FortressQuantization = {
  values: number[];
  integers: number[];
  scale: number;
};

export const FIELD_MODULUS = 4_294_967_291n;

export function decodeQ4_0(scale: number, codes: number[]): number[] {
  return codes.map((code) => {
    if (!Number.isInteger(code) || code < 0 || code > 15) throw new Error("Q4_0 code must be 0..15");
    return scale * (code - 8);
  });
}

export function decodeExl2Mixed(input: { bits: number; scale: number; codes: number[] }[]): number[] {
  return input.flatMap(({ bits, scale, codes }) => {
    if (![2, 3, 4, 5, 6, 8].includes(bits)) throw new Error("unsupported EXL2 bitrate");
    const zero = 2 ** (bits - 1);
    return codes.map((code) => {
      if (!Number.isInteger(code) || code < 0 || code >= 2 ** bits) throw new Error("EXL2 code is out of range");
      return (code - zero) * scale;
    });
  });
}

export function quantizeFortress(values: number[], bits = 4): FortressQuantization {
  if (bits !== 4) throw new Error("the laboratory Fortress format uses four-bit weights");
  const maximum = Math.max(...values.map((value) => Math.abs(value)), Number.MIN_VALUE);
  const scale = maximum / 7;
  return {
    values: values.slice(),
    integers: values.map((value) => Math.max(-7, Math.min(7, Math.round(value / scale)))),
    scale,
  };
}

export function additiveShares(values: number[], count = 2, seed = 0x12345678): bigint[][] {
  if (count < 2) throw new Error("at least two shares are required");
  let state = seed >>> 0;
  const shares = Array.from({ length: count - 1 }, () =>
    values.map(() => {
      state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
      return BigInt(state % Number(FIELD_MODULUS));
    }),
  );
  const combined = values.map((_, index) =>
    shares.slice(0, count - 1).reduce((sum, share) => sum + share[index]!, 0n),
  );
  shares.push(values.map((value, index) => modField(BigInt(value) - combined[index]!)));
  return shares;
}

export function modField(value: bigint): bigint {
  const remainder = value % FIELD_MODULUS;
  return remainder < 0n ? remainder + FIELD_MODULUS : remainder;
}

export function combineShares(shares: bigint[][]): bigint[] {
  if (shares.length < 2) throw new Error("cannot combine fewer than two shares");
  return shares[0]!.map((_, index) =>
    shares.reduce((sum, share) => modField(sum + share[index]!), 0n),
  );
}

export function procedural3inst(value: number): number {
  let product = (Math.imul(value, 89_226_354) + 64_248_484) >>> 0;
  product = ((product & 0x8fff8fff) ^ 0x3b603b60) >>> 0;
  return fp16(product & 0xffff) + fp16(product >>> 16);
}

export function fp16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

export function quantizationError(source: number[], recovered: number[]): number {
  return source.reduce((maximum, value, index) => Math.max(maximum, Math.abs(value - recovered[index]!), 0), 0);
}
