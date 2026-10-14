import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "دستیار هوشمند علوم تجربی هفتم | Unified RAG",
  description: "سامانه پاسخگویی به سوالات کتاب علوم تجربی پایه هفتم با استفاده از هوش مصنوعی و بازیابی مستندات",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="fa" dir="rtl" className="dark">
      <body className="min-h-screen flex flex-col">{children}</body>
    </html>
  )
}
