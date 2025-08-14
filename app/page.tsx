"use client"

import { useState } from "react"
import { BottomNavigation } from "@/components/bottom-navigation"
import MediMate from "@/components/modules/medi-mate"
import MediMood from "@/components/modules/medi-mood"
import PulseChain from "@/components/modules/pulse-chain"
import DocMate from "@/components/modules/doc-mate"
import { ThemeToggle } from "@/components/theme-toggle"
import { WelcomeScreen } from "@/components/welcome-screen"

export default function Home() {
  const [activeTab, setActiveTab] = useState("medimate")

  return (
    <div className="relative min-h-screen pb-16">
      <WelcomeScreen />
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md p-4 border-b border-gray-100 dark:border-gray-800 flex justify-between items-center">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-teal-500 to-blue-500 bg-clip-text text-transparent">
          HealthX Hub
        </h1>
        <ThemeToggle />
      </header>

      <div className="p-4">
        {activeTab === "medimate" && <MediMate />}
        {activeTab === "medimood" && <MediMood />}
        {activeTab === "pulsechain" && <PulseChain />}
        {activeTab === "docmate" && <DocMate />}
      </div>

      <BottomNavigation activeTab={activeTab} setActiveTab={setActiveTab} />
    </div>
  )
}
