"use client"

import type { ReactNode } from "react"
import { MessageSquare, SmilePlus, AlertTriangle, Stethoscope, UploadCloud } from "lucide-react"
import { cn } from "@/lib/utils"

interface BottomNavigationProps {
  activeTab: string
  setActiveTab: (tab: string) => void
}

export function BottomNavigation({ activeTab, setActiveTab }: BottomNavigationProps) {
  const tabs: { id: string; label: ReactNode; icon: any; color: string }[] = [
    {
      id: "medimate",
      label: "MediMate",
      icon: MessageSquare,
      color: "from-teal-500 to-blue-500",
    },
    {
      id: "medimood",
      label: "MediMood",
      icon: SmilePlus,
      color: "from-purple-500 to-pink-500",
    },
    {
      id: "pulsechain",
      label: "PulseChain",
      icon: AlertTriangle,
      color: "from-red-500 to-orange-500",
    },
    {
      id: "docmate",
      label: "DocMate",
      icon: Stethoscope,
      color: "from-blue-500 to-indigo-500",
    },
    {
      id: "xray-analyser",
      label: (
        <>
          Xray &amp; MRI
          <br />
          analyser
        </>
      ),
      icon: UploadCloud,
      color: "from-blue-600 to-blue-400",
    },
  ]

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 shadow-lg rounded-t-xl z-50">
      <div className="flex justify-around items-center h-16">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "flex flex-col items-center justify-center w-full h-full transition-all duration-300",
              activeTab === tab.id ? "text-transparent bg-clip-text bg-gradient-to-r " + tab.color : "text-gray-500",
            )}
          >
            <tab.icon className={cn("h-5 w-5 mb-1", activeTab === tab.id ? "stroke-current" : "stroke-gray-500")} />
            <span className="text-xs font-medium">{tab.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
