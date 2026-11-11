/** 75.4 -> "1:15" */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** 0.0227 -> "2.3%" */
export function pct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

/** 0.96958 -> "0.970" */
export function score(x: number): string {
  return x.toFixed(3);
}

export function seconds(ms: number): string {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** "call_0079:3" -> { callId: "call_0079", turnId: 3 } */
export function parseChunkId(id: string): { callId: string; turnId: number } {
  const i = id.lastIndexOf(":");
  return { callId: id.slice(0, i), turnId: Number(id.slice(i + 1)) };
}

export const SPLIT_LABEL: Record<string, string> = { train: "Train", dev: "Dev", test: "Test", upload: "Uploaded" };
export const SOURCE_LABEL: Record<string, string> = { template: "Billing template", multiwoz: "Bookings (MultiWOZ)", upload: "Uploaded" };
export const SPEAKER_LABEL: Record<string, string> = { agent: "Agent", customer: "Customer" };
