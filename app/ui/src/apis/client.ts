// Backend HTTP client. The UI is a thin consumer of the FastAPI surface.
// Domain-specific clients (videos, labels, policies, search, monitoring) build
// on this base.

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}
