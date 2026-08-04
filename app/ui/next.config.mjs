/** @type {import('next').NextConfig} */
// Proxy /api to the backend so the browser only needs the UI port (3000).
// With NEXT_PUBLIC_API_BASE unset/empty the client fetches same-origin /api/*,
// which Next rewrites to the backend below (override via API_PROXY_TARGET).
const API = process.env.API_PROXY_TARGET || "http://localhost:8000";
const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};
export default nextConfig;
