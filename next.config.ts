import type { NextConfig } from "next";
const config: NextConfig = {
  poweredByHeader: false,
  devIndicators: false,
  turbopack: { root: process.cwd() },
  // The landing page moved from /site to /; the lab moved from / to /lab.
  async redirects() {
    return [{ source: "/site", destination: "/", permanent: true }];
  },
};
export default config;
