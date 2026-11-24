/**
 * Drawing for the live demonstration's screens. Each function tolerates a missing 2D context, so the
 * show still runs where canvases cannot be drawn.
 */

const INK = "#e5edf5";
const DIM = "#8ea2b6";
const PAPER = "#0a1018";

function resize(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D | null {
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  return canvas.getContext("2d");
}

/** RGB bytes, one pixel per canvas pixel. */
export function paintPixels(canvas: HTMLCanvasElement, rgb: ArrayLike<number>, width: number, height: number): void {
  const context = resize(canvas, width, height);
  if (!context) return;
  const image = context.createImageData(width, height);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    image.data[pixel * 4] = rgb[pixel * 3]!;
    image.data[pixel * 4 + 1] = rgb[pixel * 3 + 1]!;
    image.data[pixel * 4 + 2] = rgb[pixel * 3 + 2]!;
    image.data[pixel * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

export function paintBlank(canvas: HTMLCanvasElement, width: number, height: number): void {
  const context = resize(canvas, width, height);
  if (!context) return;
  context.fillStyle = PAPER;
  context.fillRect(0, 0, width, height);
}

/** A worker's numbers as rows of hexadecimal: what it holds when the output is text. */
export function paintNumbers(canvas: HTMLCanvasElement, values: ArrayLike<number>, bad: boolean): void {
  const context = resize(canvas, 256, 160);
  if (!context) return;
  context.fillStyle = PAPER;
  context.fillRect(0, 0, 256, 160);
  context.font = "600 22px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textBaseline = "top";
  context.fillStyle = bad ? "#fdba74" : "#7dd3fc";
  for (let row = 0; row < 6; row += 1) {
    const cells: string[] = [];
    for (let col = 0; col < 4; col += 1) cells.push((values[(row * 4 + col) % Math.max(values.length, 1)] ?? 0).toString(16).padStart(4, "0"));
    context.fillText(cells.join(" "), 10, 8 + row * 25);
  }
}

/** The prompt and the answer so far, as they appear on the device's screen in the 3D view. */
export function paintText(canvas: HTMLCanvasElement, prompt: string, answer: string): void {
  const width = 640;
  const height = 400;
  const context = resize(canvas, width, height);
  if (!context) return;
  context.fillStyle = PAPER;
  context.fillRect(0, 0, width, height);
  context.textBaseline = "top";
  context.font = "600 22px system-ui, sans-serif";
  context.fillStyle = DIM;
  context.fillText(prompt, 24, 20, width - 48);
  context.font = "500 30px system-ui, sans-serif";
  context.fillStyle = INK;
  const lines: string[] = [];
  let line = "";
  for (const word of answer.split(" ")) {
    const next = line ? `${line} ${word}` : word;
    if (line && context.measureText(next).width > width - 48) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  lines.push(`${line}▌`);
  // Keep the newest lines in view once the screen is full.
  lines.slice(-8).forEach((text, index) => context.fillText(text, 24, 64 + index * 40));
}

/** A titled card of text for a screen in a lab's 3D view. */
export function paintCard(canvas: HTMLCanvasElement, title: string, text: string, tone: "good" | "accent" | "warn"): void {
  const width = 512;
  const height = 320;
  const context = resize(canvas, width, height);
  if (!context) return;
  context.fillStyle = PAPER;
  context.fillRect(0, 0, width, height);
  context.textBaseline = "top";
  context.font = "700 30px system-ui, sans-serif";
  context.fillStyle = tone === "good" ? "#86efac" : tone === "warn" ? "#fdba74" : "#7dd3fc";
  context.fillText(title.toUpperCase(), 22, 20, width - 44);
  // Short values are drawn large; long ones shrink, then wrap.
  const longest = Math.max(1, ...text.split("\n").map((line) => line.length));
  const size = Math.max(36, Math.min(72, Math.floor((width - 44) / (0.62 * longest))));
  context.font = `600 ${size}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  context.fillStyle = INK;
  const lines: string[] = [];
  for (const paragraph of text.split("\n")) {
    let line = "";
    for (const word of paragraph.split(" ")) {
      const next = line ? `${line} ${word}` : word;
      if (line && context.measureText(next).width > width - 44) {
        lines.push(line);
        line = word;
      } else {
        line = next;
      }
    }
    lines.push(line);
  }
  const pitch = Math.round(size * 1.25);
  lines.slice(0, Math.max(1, Math.floor(236 / pitch))).forEach((entry, index) => context.fillText(entry, 22, 76 + index * pitch, width - 44));
}

/** Sealed bytes as violet static: what a relay or a network observer has in front of it. */
export function paintCipher(canvas: HTMLCanvasElement, bytes: ArrayLike<number> | null): void {
  const width = 32;
  const height = 20;
  const context = resize(canvas, width, height);
  if (!context) return;
  const image = context.createImageData(width, height);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const value = bytes && bytes.length > 0 ? bytes[pixel % bytes.length]! & 255 : 0;
    image.data[pixel * 4] = bytes ? 40 + value * 0.6 : 10;
    image.data[pixel * 4 + 1] = bytes ? 20 + value * 0.35 : 16;
    image.data[pixel * 4 + 2] = bytes ? 90 + value * 0.65 : 24;
    image.data[pixel * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}
