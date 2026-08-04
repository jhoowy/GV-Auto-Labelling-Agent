"use client";
// DB browser: pick a table (name + row count), then browse a paginated, text-
// filterable view of its rows. Vector columns arrive pre-summarized as
// "vector(n)"; JSON/object cells render readably.
import { useEffect, useState } from "react";
import { getDbTable, listDbTables } from "../../apis/client";
import type { DbRow } from "../../apis/types";
import { useAsync } from "../../lib/useAsync";
import { AsyncState } from "../../components/ui";

const PAGE_SIZE = 25;

export default function DbBrowser() {
  const tables = useAsync(() => listDbTables(), []);
  const [table, setTable] = useState<string | null>(null);

  const list = tables.data ?? [];

  return (
    <div className="grid" style={{ gap: 16 }}>
      <h1>DB Browser</h1>
      <div className="two-col">
        <div className="card">
          <h3>Tables</h3>
          <AsyncState
            loading={tables.loading}
            error={tables.error}
            empty={list.length === 0}
            emptyText="No tables reported."
          >
            <div className="list">
              {list.map((t) => (
                <button
                  key={t.name}
                  className={`list-item${table === t.name ? " active" : ""}`}
                  onClick={() => setTable(t.name)}
                >
                  <div className="row spread">
                    <span className="mono">{t.name}</span>
                    <span className="muted small">{t.count.toLocaleString()}</span>
                  </div>
                </button>
              ))}
            </div>
          </AsyncState>
        </div>

        {table ? (
          <TableView name={table} />
        ) : (
          <div className="card state">Select a table to browse its rows.</div>
        )}
      </div>
    </div>
  );
}

function TableView({ name }: { name: string }) {
  const [input, setInput] = useState("");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(1);

  // reset when switching tables
  useEffect(() => {
    setInput("");
    setQ("");
    setPage(1);
  }, [name]);

  // debounce filter
  useEffect(() => {
    const t = setTimeout(() => {
      setQ(input.trim());
      setPage(1);
    }, 300);
    return () => clearTimeout(t);
  }, [input]);

  const { data, loading, error } = useAsync(
    () => getDbTable(name, { page, page_size: PAGE_SIZE, q }),
    [name, page, q],
  );

  const columns = data?.columns ?? [];
  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="card">
      <div className="row spread" style={{ marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>
          <span className="mono">{name}</span>{" "}
          {total > 0 && <span className="muted small">({total} rows)</span>}
        </h3>
        <div style={{ width: 240 }}>
          <input
            type="search"
            placeholder="Filter rows (q)…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
        </div>
      </div>

      <AsyncState
        loading={loading}
        error={error}
        empty={rows.length === 0}
        emptyText={q ? `No rows match “${q}”.` : "Table is empty."}
      >
        <div style={{ overflowX: "auto" }}>
          <table className="db-table">
            <thead>
              <tr>
                {columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i}>
                  {columns.map((c, j) => (
                    <td key={c}>
                      <Cell value={cellValue(row, c, j)} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </AsyncState>

      {pageCount > 1 && (
        <div className="row" style={{ justifyContent: "center", gap: 12, marginTop: 12 }}>
          <button
            className="btn"
            disabled={page <= 1 || loading}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            ← Prev
          </button>
          <span className="muted small">
            Page {page} / {pageCount}
          </span>
          <button
            className="btn"
            disabled={page >= pageCount || loading}
            onClick={() => setPage((p) => p + 1)}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}

// Rows may be positional arrays or column-keyed objects.
function cellValue(row: DbRow, col: string, idx: number): unknown {
  if (Array.isArray(row)) return row[idx];
  return (row as Record<string, unknown>)[col];
}

function Cell({ value }: { value: unknown }) {
  if (value == null) return <span className="muted">null</span>;
  if (typeof value === "object") {
    return (
      <details>
        <summary className="mono small">
          {Array.isArray(value) ? `[${value.length}]` : "{…}"}
        </summary>
        <pre className="cell-json">{JSON.stringify(value, null, 2)}</pre>
      </details>
    );
  }
  const s = String(value);
  return <span className="mono small">{s.length > 200 ? `${s.slice(0, 200)}…` : s}</span>;
}
