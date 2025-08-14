"use client"

import { motion } from "framer-motion"

type Mood = {
  emoji: string
  label: string
  color: string
  gradient: string
}

const moods: Mood[] = [
  {
    emoji: "😊",
    label: "Happy",
    color: "bg-yellow-500",
    gradient: "from-yellow-400 to-orange-300",
  },
  {
    emoji: "😌",
    label: "Calm",
    color: "bg-blue-500",
    gradient: "from-blue-400 to-teal-300",
  },
  {
    emoji: "😐",
    label: "Neutral",
    color: "bg-gray-500",
    gradient: "from-gray-400 to-slate-300",
  },
  {
    emoji: "😔",
    label: "Sad",
    color: "bg-indigo-500",
    gradient: "from-indigo-400 to-purple-300",
  },
  {
    emoji: "😡",
    label: "Angry",
    color: "bg-red-500",
    gradient: "from-red-400 to-pink-300",
  },
  {
    emoji: "😰",
    label: "Anxious",
    color: "bg-orange-500",
    gradient: "from-orange-400 to-amber-300",
  },
  {
    emoji: "😴",
    label: "Tired",
    color: "bg-purple-500",
    gradient: "from-purple-400 to-indigo-300",
  },
  {
    emoji: "🤗",
    label: "Grateful",
    color: "bg-green-500",
    gradient: "from-green-400 to-emerald-300",
  },
  {
    emoji: "🥳",
    label: "Excited",
    color: "bg-pink-500",
    gradient: "from-pink-400 to-rose-300",
  },
  {
    emoji: "😵",
    label: "Overwhelmed",
    color: "bg-violet-500",
    gradient: "from-violet-400 to-purple-300",
  },
  {
    emoji: "🤔",
    label: "Confused",
    color: "bg-amber-500",
    gradient: "from-amber-400 to-yellow-300",
  },
  {
    emoji: "😤",
    label: "Frustrated",
    color: "bg-rose-500",
    gradient: "from-rose-400 to-red-300",
  },
]

interface MoodPickerProps {
  selectedMood: Mood | null
  setSelectedMood: (mood: Mood | null) => void
}

export function MoodPicker({ selectedMood, setSelectedMood }: MoodPickerProps) {
  return (
    <div className="grid grid-cols-4 gap-4">
      {moods.map((mood) => (
        <motion.button
          key={mood.label}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setSelectedMood(mood)}
          className={`
            flex flex-col items-center justify-center p-3 rounded-lg
            ${
              selectedMood?.label === mood.label
                ? `bg-gradient-to-br ${mood.gradient} text-white ring-2 ring-offset-2 ring-${mood.color.replace("bg-", "")}`
                : "bg-white hover:bg-gray-50"
            }
            transition-all duration-200
          `}
        >
          <span className="text-3xl mb-1">{mood.emoji}</span>
          <span className="text-xs font-medium">{mood.label}</span>
        </motion.button>
      ))}
    </div>
  )
}
