"use client"

import type React from "react"
import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2, Send, Video, BookOpen, Music } from "lucide-react"

type Guide = {
  id: string
  title: string
  description: string
  url: string
}

const readingGuides: Guide[] = [
  {
    id: "1",
    title: "AI for Beginners",
    description: "Start your AI journey with this beginner-friendly guide.",
    url: "https://example.com/ai-beginners",
  },
  {
    id: "2",
    title: "Machine Learning Basics",
    description: "Learn the fundamentals of machine learning step by step.",
    url: "https://example.com/ml-basics",
  },
]

const videoGuides: Guide[] = [
  {
    id: "1",
    title: "Intro to AI",
    description: "Watch this beginner-friendly video introduction to AI.",
    url: "https://www.youtube.com/watch?v=example1",
  },
  {
    id: "2",
    title: "ML Explained",
    description: "A comprehensive video on machine learning concepts.",
    url: "https://www.youtube.com/watch?v=example2",
  },
]

const musicGuides: Guide[] = [
  {
    id: "1",
    title: "LoFi Study Beats",
    description: "Relax and focus with AI-generated LoFi beats.",
    url: "https://open.spotify.com/track/example1",
  },
  {
    id: "2",
    title: "Calm Focus",
    description: "Instrumental music to help you concentrate.",
    url: "https://open.spotify.com/track/example2",
  },
]

export default function Guides() {
  const [loading, setLoading] = useState(false)
  const [input, setInput] = useState("")

  const handleAskAI = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
      setInput("")
    }, 2000)
  }

  return (
    <div className="max-w-4xl mx-auto p-4 space-y-6">
      {/* Search Bar */}
      <form onSubmit={handleAskAI} className="flex gap-2">
        <Input
          placeholder="Ask AI for personalized guidance..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <Button type="submit" disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </Button>
      </form>

      {/* Tabs Section */}
      <Tabs defaultValue="reading" className="w-full">
        <TabsList className="grid grid-cols-3 w-full">
          <TabsTrigger value="reading" className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" /> Reading
          </TabsTrigger>
          <TabsTrigger value="videos" className="flex items-center gap-2">
            <Video className="h-4 w-4" /> Videos
          </TabsTrigger>
          <TabsTrigger value="music" className="flex items-center gap-2">
            <Music className="h-4 w-4" /> Music
          </TabsTrigger>
        </TabsList>

        {/* Reading Tab */}
        <TabsContent value="reading" className="space-y-4 mt-4">
          {readingGuides.map((item) => (
            <Card key={item.id}>
              <CardContent className="p-4 flex flex-col space-y-2">
                <h3 className="font-semibold">{item.title}</h3>
                <p className="text-sm text-gray-600">{item.description}</p>
                <Button asChild className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">
                  <a href={item.url} target="_blank" rel="noopener noreferrer">
                    Read Now
                  </a>
                </Button>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        {/* Videos Tab */}
        <TabsContent value="videos" className="space-y-4 mt-4">
          {videoGuides.map((item) => (
            <Card key={item.id}>
              <CardContent className="p-4 flex flex-col space-y-2">
                <h3 className="font-semibold">{item.title}</h3>
                <p className="text-sm text-gray-600">{item.description}</p>
                <Button asChild className="w-full mt-3 bg-gradient-to-r from-blue-500 to-cyan-500">
                  <a href={item.url} target="_blank" rel="noopener noreferrer">
                    Watch Now
                  </a>
                </Button>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        {/* Music Tab */}
        <TabsContent value="music" className="space-y-4 mt-4">
          {musicGuides.map((item) => (
            <Card key={item.id}>
              <CardContent className="p-4 flex flex-col space-y-2">
                <h3 className="font-semibold">{item.title}</h3>
                <p className="text-sm text-gray-600">{item.description}</p>
                <Button asChild className="w-full mt-3 bg-gradient-to-r from-green-500 to-emerald-500">
                  <a href={item.url} target="_blank" rel="noopener noreferrer">
                    Listen Now
                  </a>
                </Button>
              </CardContent>
            </Card>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  )
}
