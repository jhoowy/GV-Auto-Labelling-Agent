"use client";
import { useCallback, useEffect, useState } from "react";

interface AsyncResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

// Runs an async fetcher on mount and whenever `deps` change. Errors are
// captured so pages can render an error state instead of crashing — important
// because several backend endpoints are still stubs (500 / NotImplementedError).
export function useAsync<T>(
  fetcher: () => Promise<T>,
  deps: unknown[],
): AsyncResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fetcher()
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, loading, error, reload };
}
