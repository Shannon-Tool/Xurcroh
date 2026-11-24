/** A node label projected to screen pixels. `depth` is 0 at the camera and 1 at the far plane. */
export interface LabelPoint {
  id: string;
  text: string;
  x: number;
  y: number;
  depth: number;
  visible: boolean;
  active: boolean;
}

interface Rect { left: number; right: number; top: number; bottom: number }

// Labels are anchored at their bottom centre; the size is estimated so this stays independent of the DOM.
function rectOf(point: LabelPoint): Rect {
  const width = point.text.length * 6.2 + 16;
  return { left: point.x - width / 2, right: point.x + width / 2, top: point.y - 22, bottom: point.y };
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

/** Picks the labels to show: the active step's first, then nearest first, skipping any that would overlap. */
export function declutter(points: LabelPoint[], pinnedId?: string | null): LabelPoint[] {
  const rank = (point: LabelPoint) => (point.id === pinnedId ? 0 : point.active ? 1 : 2);
  const ordered = points.filter((point) => point.visible).sort((a, b) => rank(a) - rank(b) || a.depth - b.depth);
  const taken: Rect[] = [];
  const shown: LabelPoint[] = [];
  for (const point of ordered) {
    const rect = rectOf(point);
    if (taken.some((other) => overlaps(rect, other))) continue;
    taken.push(rect);
    shown.push(point);
  }
  return shown;
}
