"use client"

import { format } from "date-fns"
import { Card, CardContent } from "@/components/ui/card"
import { motion, AnimatePresence } from "framer-motion"

type Mood = {
  emoji: string
  label: string
  color: string
  gradient: string
}

type Entry = {
  id: string
  date: Date
  mood: Mood
  text: string
}

interface JournalEntriesProps {
  entries: Entry[]
}

export function JournalEntries({ entries }: JournalEntriesProps) {
  if (entries.length === 0) {
    return (
      <Card className="bg-gray-50">
        <CardContent className="p-8 text-center text-gray-500">
          No journal entries yet. Start by recording your mood.
        </CardContent>
      </Card>
    )
  }

  return (
    <AnimatePresence>
      <div className="space-y-4">
        {entries.map((entry) => (
          <motion.div
            key={entry.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.3 }}
          >
            <Card
              className={`bg-gradient-to-r ${entry.mood.gradient.replace("from-", "from-opacity-10 ").replace("to-", "to-opacity-5 ")} border-l-4 border-l-${entry.mood.color.replace("bg-", "")}`}
            >
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <div
                    className={`flex items-center justify-center rounded-full ${entry.mood.color} w-10 h-10 text-white text-xl flex-shrink-0`}
                  >
                    {entry.mood.emoji}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium">{entry.mood.label}</h4>
                      <span className="text-xs text-gray-500">{format(entry.date, "PPP p")}</span>
                    </div>
                    <p className="text-gray-700 text-sm whitespace-pre-line">{entry.text}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </AnimatePresence>
  )
}
