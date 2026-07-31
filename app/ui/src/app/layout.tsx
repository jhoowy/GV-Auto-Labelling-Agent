import type { ReactNode } from "react";

export const metadata = {
  title: "Video Labelling — Content Moderation PoC",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <nav style={{ display: "flex", gap: 16, padding: 12, borderBottom: "1px solid #ddd" }}>
          <a href="/viewer">Data Viewer</a>
          <a href="/monitoring">Monitoring</a>
        </nav>
        <main style={{ padding: 16 }}>{children}</main>
      </body>
    </html>
  );
}
