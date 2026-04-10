import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow connections from the provided LAN IP for HMR
  allowedDevOrigins: ['192.168.1.17', 'localhost'],
};

export default nextConfig;
