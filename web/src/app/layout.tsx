import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MLX-Tune-GUI",
  description: "Advanced Mac-native visual orchestrator for MLX fine-tuning",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>
        <div className="app-container">
          {children}
        </div>
      </body>
    </html>
  );
}
