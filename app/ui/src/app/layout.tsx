import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Video Labelling — Content Moderation PoC",
  description: "Operator console for the agentic video labelling PoC.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="topnav">
          <a className="brand" href="/">
            Video Labelling
          </a>
          <a href="/viewer">Data Viewer</a>
          <a href="/policy">Policy</a>
          <a href="/db">DB Browser</a>
        </nav>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
