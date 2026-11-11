import { Fragment } from "react";

/** Plain-language names for the placeholder types the redactor writes. */
export const REDACTION_LABEL: Record<string, string> = {
  PERSON: "name",
  CARD: "card number",
  PHONE: "phone",
  EMAIL: "email",
  SIN: "SIN",
  ACCOUNT: "account",
};

const PLACEHOLDER = /\[([A-Z_]+)\]/g;

/**
 * Renders transcript text with each [TYPE] placeholder as a solid redaction bar.
 * Screen readers hear "redacted name", not "left bracket PERSON right bracket".
 */
export function RedactedText({ text }: { text: string }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(PLACEHOLDER)) {
    const i = m.index ?? 0;
    if (i > last) parts.push(text.slice(last, i));
    const type = m[1] ?? "";
    const label = REDACTION_LABEL[type] ?? type.toLowerCase();
    parts.push(
      <span className="redaction" key={i} title={`Redacted ${label}`}>
        <span className="sr-only">redacted </span>
        {label}
      </span>,
    );
    last = i + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <Fragment>{parts}</Fragment>;
}
