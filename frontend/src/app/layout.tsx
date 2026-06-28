import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "C-RAG Assistant | Corrective Retrieval-Augmented Generation",
  description: "A Corrective RAG (C-RAG) visual tutor system for machine learning concepts, running on a local Ollama stack with LangGraph.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
