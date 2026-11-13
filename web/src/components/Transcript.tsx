import { useEffect, useRef } from "react";
import type { CallDetail } from "../api/client";
import { clock, SPEAKER_LABEL } from "../lib/format";
import { RedactedText } from "./RedactedText";

/** A redacted transcript. `citedTurn` is highlighted like a marker on paper and scrolled into view. */
export function Transcript({ call, citedTurn, scrollRoot }: {
  call: CallDetail;
  citedTurn?: number;
  scrollRoot?: React.RefObject<HTMLElement | null>;
}) {
  const cited = useRef<HTMLLIElement>(null);
  useEffect(() => {
    const el = cited.current;
    if (!el) return;
    const root = scrollRoot?.current;
    if (root) root.scrollTop = el.offsetTop - root.clientHeight / 3;
    else el.scrollIntoView?.({ block: "center" });
  }, [call.call_id, citedTurn, scrollRoot]);

  return (
    <ol className="transcript" aria-label={`Transcript of ${call.call_id}`}>
      {call.turns.map((t) => {
        const isCited = t.turn_id === citedTurn;
        return (
          <li
            key={t.turn_id}
            id={`turn-${t.turn_id}`}
            ref={isCited ? cited : undefined}
            className={isCited ? "turn is-cited" : "turn"}
            aria-current={isCited ? "true" : undefined}
          >
            <span className="turn-time">{clock(t.start_sec)}</span>
            <span className={t.speaker === "agent" ? "turn-speaker is-agent" : "turn-speaker"}>
              {SPEAKER_LABEL[t.speaker]}
              {isCited && <span className="sr-only">, cited turn</span>}
            </span>
            <span className="turn-text">
              <RedactedText text={t.text} />
            </span>
          </li>
        );
      })}
    </ol>
  );
}
