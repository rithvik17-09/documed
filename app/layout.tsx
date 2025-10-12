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
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} min-h-screen bg-background text-foreground`}>
        <SessionProviderWrapper>
          <ThemeProvider 
            attribute="class" 
            defaultTheme="light" 
            enableSystem
            disableTransitionOnChange={false}
          >
            <main className="min-h-screen">
              {children}
            </main>
          </ThemeProvider>
        </SessionProviderWrapper>
      </body>
    </html>
  );
}

