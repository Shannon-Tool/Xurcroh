export type Matrix = number[][];
export type Vector = number[];

export function matMul(a: Matrix, b: Matrix): Matrix {
  if (a.length === 0 || b.length === 0 || a[0]!.length !== b.length) {
    throw new Error("matrix dimensions do not align");
  }
  const columns = b[0]!.length;
  return a.map((row) => {
    if (row.length !== b.length) throw new Error("ragged matrix");
    return Array.from({ length: columns }, (_, column) =>
      row.reduce((sum, value, index) => sum + value * b[index]![column]!, 0),
    );
  });
}

export function transpose(matrix: Matrix): Matrix {
  return matrix[0]!.map((_, column) => matrix.map((row) => row[column]!));
}

export function rotate2d(angle: number): Matrix {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [[cosine, -sine], [sine, cosine]];
}

export function conjugateLinear(weight: Matrix, inputBasis: Matrix, outputBasis: Matrix): Matrix {
  return matMul(matMul(outputBasis, weight), transpose(inputBasis));
}

export function fwht(input: Vector): { stages: Vector[]; output: Vector } {
  const dimension = input.length;
  if (dimension < 1 || (dimension & (dimension - 1)) !== 0) {
    throw new Error("FWHT dimension must be a power of two");
  }
  const stages = [input.slice()];
  let block = 1;
  while (block < dimension) {
    const previous = stages.at(-1)!.slice();
    for (let start = 0; start < dimension; start += block * 2) {
      for (let offset = 0; offset < block; offset += 1) {
        const left = previous[start + offset]!;
        const right = previous[start + block + offset]!;
        previous[start + offset] = left + right;
        previous[start + block + offset] = left - right;
      }
    }
    stages.push(previous);
    block *= 2;
  }
  const scale = Math.sqrt(dimension);
  const output = stages.at(-1)!.map((value) => value / scale);
  stages[stages.length - 1] = output;
  return { stages, output };
}

export function maxError(a: Matrix, b: Matrix): number {
  return a.reduce(
    (maximum, row, i) => Math.max(maximum, ...row.map((value, j) => Math.abs(value - b[i]![j]!))),
    0,
  );
}

export function eulerStep(latent: Matrix, velocity: Matrix, delta: number): Matrix {
  return latent.map((row, i) => row.map((value, j) => value - delta * velocity[i]![j]!));
}

export function frameDeltaStep(
  state: Matrix,
  keys: Matrix,
  values: Matrix,
  queries: Matrix,
  beta: number[],
): { state: Matrix; output: Matrix } {
  const dk = keys[0]!.length;
  const identity = Array.from({ length: dk }, (_, i) =>
    Array.from({ length: dk }, (_, j) => (i === j ? 1 : 0)),
  );
  const a = Array.from({ length: dk }, () => Array(dk).fill(0));
  for (let i = 0; i < dk; i += 1) {
    for (let j = 0; j < dk; j += 1) {
      a[i]![j] = keys.reduce((sum, key, token) => sum + beta[token]! * key[i]! * key[j]!, 0) + identity[i]![j]!;
    }
  }
  const b = values[0]!.map((_, i) =>
    Array.from({ length: dk }, (_, j) =>
      values.reduce((sum, value, token) => sum + beta[token]! * value[i]! * keys[token]![j]!, 0),
    ),
  );
  const right = state.map((row, i) => row.map((value, j) => value + b[i]![j]!));
  const updated = right.map((row) => solve(a, row));
  return { state: updated, output: matMul(updated, transpose(queries)) };
}

function solve(matrix: Matrix, vector: Vector): Vector {
  const augmented = matrix.map((row, i) => [...row, vector[i]!]);
  for (let column = 0; column < matrix.length; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < matrix.length; row += 1) {
      if (Math.abs(augmented[row]![column]!) > Math.abs(augmented[pivot]![column]!)) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot]!, augmented[column]!];
    const divisor = augmented[column]![column]!;
    if (Math.abs(divisor) < 1e-12) throw new Error("singular normal equation");
    for (let row = 0; row < matrix.length; row += 1) {
      if (row === column) continue;
      const factor = augmented[row]![column]! / divisor;
      for (let j = column; j <= matrix.length; j += 1) {
        augmented[row]![j]! -= factor * augmented[column]![j]!;
      }
    }
  }
  return augmented.map((row, i) => row[matrix.length]! / augmented[i]![i]!);
}
