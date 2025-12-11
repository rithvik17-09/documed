"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { motion } from "framer-motion"
import { MoodPicker } from "./mood-picker"
import { JournalEntries } from "./journal-entries"
import { ContentSuggestions } from "./content-suggestions"
import { MoodAnalyzer } from "./mood-analyzer"

type Mood = {
  emoji: string
  label: string
  color: string
  gradient: string
}

export default function MediMood() {
  const [selectedMood, setSelectedMood] = useState<Mood | null>(null)
  const [journalText, setJournalText] = useState("")
  const [entries, setEntries] = useState<
    Array<{
      id: string
      date: Date
      mood: Mood
      text: string
    }>
  >([
    {
      id: "1",
      date: new Date(2025, 4, 29),
      mood: {
        emoji: "😊",
        label: "Happy",
        color: "bg-yellow-500",
        gradient: "from-yellow-400 to-orange-300",
      },
      text: "Had a great day today! The weather was perfect for a walk in the park.",
    },
  ])

  const [activeTab, setActiveTab] = useState("mood")
  const [bgColor, setBgColor] = useState("bg-purple-50")

  useEffect(() => {
    if (selectedMood) {
      const baseColor = selectedMood.gradient.split(" ")[0].replace("from-", "")
      setBgColor(`bg-${baseColor}-50`)
    } else {
      setBgColor("bg-purple-50")
    }
  }, [selectedMood])

  const handleSaveEntry = () => {
    if (!selectedMood || !journalText.trim()) return

    const newEntry = {
      id: Date.now().toString(),
      date: new Date(),
      mood: selectedMood,
      text: journalText,
    }

    setEntries((prev) => [newEntry, ...prev])
    setJournalText("")
    setSelectedMood(null)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.3 }}
      className="space-y-4"
    >
      <h2 className="text-xl font-semibold text-purple-700">MediMood</h2>
      <p className="text-gray-600">Track your mental wellbeing</p>

      <Tabs defaultValue="mood" className="w-full" onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 mb-4">
          <TabsTrigger value="mood">Mood Tracker</TabsTrigger>
          <TabsTrigger value="analyzer">Mood Analyzer</TabsTrigger>
          <TabsTrigger value="journal">Journal</TabsTrigger>
          <TabsTrigger value="content">Content</TabsTrigger>
        </TabsList>

        <TabsContent value="mood" className="space-y-4">
          <Card className={`transition-colors duration-500 ${bgColor}`}>
            <CardHeader>
              <CardTitle className="text-center">How are you feeling today?</CardTitle>
            </CardHeader>
            <CardContent>
              <MoodPicker selectedMood={selectedMood} setSelectedMood={setSelectedMood} />

              {selectedMood && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="mt-6 space-y-4"
                >
                  <Textarea
                    placeholder="Write about how you're feeling..."
                    className="min-h-[120px] bg-white/80"
                    value={journalText}
                    onChange={(e) => setJournalText(e.target.value)}
                  />
                  <Button
                    onClick={handleSaveEntry}
                    className={`w-full bg-gradient-to-r ${selectedMood.gradient}`}
                    disabled={!journalText.trim()}
                  >
                    Save Entry
                  </Button>
                </motion.div>
              )}
            </CardContent>
          </Card>

          {entries.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Recent Entries</h3>
              <JournalEntries entries={entries.slice(0, 3)} />
            </div>
          )}
        </TabsContent>

        <TabsContent value="analyzer" className="space-y-4">
          <MoodAnalyzer />
        </TabsContent>

        <TabsContent value="journal" className="space-y-4">
          <JournalEntries entries={entries} />
        </TabsContent>

        <TabsContent value="content" className="space-y-4">
          <ContentSuggestions mood={selectedMood} />
        </TabsContent>
      </Tabs>
    </motion.div>
  )
}
