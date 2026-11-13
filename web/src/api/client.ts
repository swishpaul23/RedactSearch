import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

export type AgentAnswerOut = components["schemas"]["AgentAnswerOut"];
export type AnswerOut = components["schemas"]["AnswerOut"];
export type AskIn = components["schemas"]["AskIn"];
export type CallDetail = components["schemas"]["CallDetail"];
export type MetricsOut = components["schemas"]["MetricsOut"];
export type UploadJob = components["schemas"]["UploadJob"];
export type Mode = AskIn["mode"];

/** An API failure with the server's stable code and its human-readable message. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const api = createClient<paths>({
  baseUrl: typeof window === "undefined" ? "" : window.location.origin,
  // Look fetch up per request (not once at import) so tests and instrumentation can wrap it.
  fetch: (request: Request) => globalThis.fetch(request),
});

type Result<T> = { data?: T; error?: unknown; response: Response };

/** Return data or throw an ApiError carrying the server's { error: { code, message } }. */
function unwrap<T>({ data, error, response }: Result<T>): T {
  if (data !== undefined && response.ok) return data;
  const body = (error ?? {}) as { error?: { code?: string; message?: string } };
  throw new ApiError(
    response.status,
    body.error?.code ?? "http_error",
    body.error?.message ?? `The server answered ${response.status}. Try again in a moment.`,
  );
}

/** Network failures (server down) become an ApiError too, so every caller handles one shape. */
export async function call<T>(request: () => Promise<Result<T>>): Promise<T> {
  let result: Result<T>;
  try {
    result = await request();
  } catch {
    throw new ApiError(0, "network", "The server could not be reached. Check that the API is running.");
  }
  return unwrap(result);
}
