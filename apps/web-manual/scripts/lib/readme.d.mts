export const START: string;
export const END: string;

export interface RecordingFacts {
  file: string;
  width: number;
  height: number;
  bytes: number;
  stored: number;
  frames: number;
  seconds: number;
  controls: string[];
  notes: string[];
}

export interface RecordingEntry {
  id: string;
  group: string;
  file: string;
  title: string;
  caption: string;
  page: string;
  lead?: boolean;
  /** The gallery stage this recording shows, if it shows one whole. */
  stage?: string;
  /** The chapter diagram this recording shows. */
  diagram?: string;
}

export function plainLabel(label: string): string;
export function renderSection(recordings: RecordingEntry[], groups: { id: string; title: string; lead: string }[], index: Record<string, RecordingFacts>): string;
export function placeSection(readme: string, section: string, after?: string): string;
