import { ApiError } from "../api/client";

export function ErrorBanner({ error, onRetry, title = "This could not be loaded" }: {
  error: unknown;
  onRetry?: () => void;
  title?: string;
}) {
  const message = error instanceof ApiError ? error.message : "Something unexpected went wrong. Try again.";
  return (
    <div className="banner banner-error" role="alert">
      <div className="banner-text">
        <p className="banner-title">{title}</p>
        <p>{message}</p>
      </div>
      {onRetry && (
        <button type="button" className="btn btn-sm" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="empty">
      <p className="empty-title">{title}</p>
      {children}
    </div>
  );
}

/** Placeholder lines while content loads (shown only for waits over ~1 s in practice). */
export function SkeletonLines({ lines = 3, label }: { lines?: number; label: string }) {
  return (
    <div role="status" aria-label={label} style={{ display: "grid", gap: 10, padding: 16 }}>
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} className="skeleton" style={{ width: `${92 - ((i * 17) % 40)}%` }} />
      ))}
    </div>
  );
}
