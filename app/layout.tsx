import React from "react"
import { ThemeProvider } from "@/components/theme-provider"
import "./globals.css"
import { Inter } from "next/font/google"
import SessionProviderWrapper from "../components/session-provider-wrapper"


const inter = Inter({ subsets: ["latin"] })


export const metadata = {
  title: "Documed",
  description: "Your personal healthcare assistant",
  generator: 'v0.dev'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <SessionProviderWrapper>
          <ThemeProvider attribute="class" defaultTheme="light">
            <main className="min-h-screen bg-gradient-to-b from-blue-50 to-green-50">{children}</main>
          </ThemeProvider>
        </SessionProviderWrapper>
      </body>
    </html>
  );
}

