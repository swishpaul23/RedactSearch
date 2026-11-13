import { useEffect, useRef, useState } from "react";

/** "/" focuses the page's primary search unless the user is already typing somewhere. */
export function useSlashFocus(target: React.RefObject<HTMLElement | null>) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName))) return;
      e.preventDefault();
      target.current?.focus();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [target]);
}

export function useDocumentTitle(title: string) {
  useEffect(() => {
    document.title = title ? `${title} | RedactSearch` : "RedactSearch";
  }, [title]);
}

/** Wraps localStorage: per-viewer conveniences only, and it can throw or be empty. */
export function usePersistentState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      return raw === null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });
  const set = (v: T) => {
    setValue(v);
    try {
      window.localStorage.setItem(key, JSON.stringify(v));
    } catch {
      /* storage unavailable: keep the in-memory value */
    }
  };
  return [value, set];
}

/** Elapsed whole seconds while `running`, for long waits on the answer model. */
export function useElapsed(running: boolean): number {
  const [secs, setSecs] = useState(0);
  const start = useRef(0);
  useEffect(() => {
    if (!running) return;
    start.current = Date.now();
    setSecs(0);
    const id = window.setInterval(() => setSecs(Math.floor((Date.now() - start.current) / 1000)), 1000);
    return () => window.clearInterval(id);
  }, [running]);
  return secs;
}
