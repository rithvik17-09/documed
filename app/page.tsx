
"use client"

import { useState } from "react"
import { useSession, signOut } from "next-auth/react"
import { useRouter } from "next/navigation"
import { useEffect } from "react"
import { BottomNavigation } from "@/components/bottom-navigation"
import MediMate from "@/components/modules/medi-mate"
import MediMood from "@/components/modules/medi-mood"
import PulseChain from "@/components/modules/pulse-chain"
import DocMate from "@/components/modules/doc-mate"
import XrayAnalyser from "@/components/modules/xray-analyser/index.lazy"
import { ThemeToggle } from "@/components/theme-toggle"
import { WelcomeScreen } from "@/components/welcome-screen"

export default function Home() {
  const [activeTab, setActiveTab] = useState("medimate")
  const { data: session, status } = useSession()
  const router = useRouter()

  useEffect(() => {
    if (status === "loading") return // Still loading
    if (!session) {
      router.push("/auth")
      return
    }
  }, [session, status, router])

  // Show loading state while checking authentication
  if (status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-teal-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  // Don't render the main content if not authenticated
  if (!session) {
    return null
  }

  return (
    <div className="relative min-h-screen pb-16 bg-gradient-to-b from-blue-50 to-green-50 dark:from-gray-900 dark:to-gray-800 transition-colors duration-300">
      <WelcomeScreen />
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md p-4 border-b border-gray-100 dark:border-gray-800 flex justify-between items-center transition-colors duration-300">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-teal-500 to-blue-500 bg-clip-text text-transparent">
          Documed
        </h1>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600 dark:text-gray-300">
            Welcome, {session?.user?.name || session?.user?.email}
          </span>
          <button
            onClick={async () => {
              // signOut without redirect, then navigate client-side to avoid
              // NextAuth constructing an absolute URL that may point to an
              // unavailable host (causes connection refused in some setups).
              await signOut({ redirect: false })
              router.push('/auth')
            }}
            className="text-sm text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-200 transition-colors"
          >
            Logout
          </button>
          <ThemeToggle />
        </div>
      </header>

      <div className="p-4">
  {activeTab === "medimate" && <MediMate />}
  {activeTab === "medimood" && <MediMood />}
  {activeTab === "pulsechain" && <PulseChain />}
  {activeTab === "docmate" && <DocMate />}
  {activeTab === "xray-analyser" && <XrayAnalyser />}
      </div>

      <BottomNavigation activeTab={activeTab} setActiveTab={setActiveTab} />
    </div>
  )
}

