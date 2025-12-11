"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ThumbsUp, ThumbsDown, Send } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

// Mood detection dictionary
const moodMap = {
  happy: {
    mood: "happy",
    keywords: [
      "happy",
      "joy",
      "great",
      "wonderful",
      "excited",
      "good",
      "positive",
      "amazing",
      "fantastic",
      "excellent",
      "cheerful",
      "delighted",
      "pleased",
      "content",
      "satisfied",
      "thrilled",
      "ecstatic",
      "elated",
      "jubilant",
      "blissful",
    ],
    response:
      "It's wonderful to hear you're feeling happy! 😊 Positive emotions like this are worth savoring. Consider taking a moment to appreciate what's going well in your life right now.",
    color: "bg-yellow-100 border-yellow-200",
    emoji: "😊",
    condition: "Positive mental wellbeing",
    remedy:
      "Maintain this state by practicing gratitude, engaging in activities you enjoy, and sharing your positive energy with others. Consider keeping a gratitude journal to reflect on positive experiences.",
    videos: [
      {
        title: "Positive Vibes Only",
        url: "https://www.youtube.com/embed/lTRiuFIWV54",
        thumbnail: "/placeholder.svg?height=120&width=200&text=Positive+Vibes",
      },
      {
        title: "Gratitude & Mindfulness",
        url: "https://www.youtube.com/embed/WPPPFqsECz0",
        thumbnail: "/placeholder.svg?height=120&width=200&text=Gratitude+Mindfulness",
      },
    ],
  },
  sad: {
    mood: "sad",
    keywords: [
      "sad",
      "down",
      "unhappy",
      "depressed",
      "blue",
      "gloomy",
      "miserable",
      "heartbroken",
      "disappointed",
      "upset",
      "discouraged",
      "disheartened",
      "melancholy",
      "somber",
      "sorrowful",
      "tearful",
      "downcast",
      "low",
      "glum",
      "despondent",
    ],
    response:
      "I'm sorry to hear you're feeling down. 💙 Remember that it's okay to feel sad sometimes, and these feelings won't last forever. Be gentle with yourself today.",
    color: "bg-blue-100 border-blue-200",
    emoji: "😔",
    condition: "Temporary sadness or possible mild depression",
    remedy:
      "Consider activities that have helped lift your mood in the past, such as going for a walk, listening to uplifting music, or reaching out to a supportive friend. If these feelings persist for more than two weeks or interfere with daily life, consider speaking with a mental health professional.",
    videos: [
      {
        title: "Relaxing Piano Music",
        url: "https://www.youtube.com/embed/lCOF9LN_Zxs",
        thumbnail: "/placeholder.svg?height=120&width=200&text=Relaxing+Piano",
      },
      {
        title: "How to Deal with Depression",
        url: "https://www.youtube.com/embed/MOGnCb4GRQ0",
        thumbnail: "/placeholder.svg?height=120&width=200&text=Deal+With+Depression",
      },
    ],
  },
  anxious: {
    mood: "anxious",
    keywords: [
      "anxious",
      "worried",
      "nervous",
      "stressed",
      "overwhelmed",
      "panic",
      "fear",
      "tense",
      "uneasy",
      "apprehensive",
      "concerned",
      "distressed",
      "frightened",
      "jittery",
      "on edge",
      "restless",
      "troubled",
      "unsettled",
      "agitated",
      "alarmed",
    ],
    response:
      "I notice you're feeling anxious. 🌬️ Remember that your feelings are valid, but they don't define you. Let's take a moment to breathe together.",
    color: "bg-pink-100 border-pink-200",
    emoji: "😟",
    condition: "Anxiety or stress response",
    remedy:
      "Try some deep breathing exercises: breathe in for 4 counts, hold for 4, and exhale for 6. Grounding techniques can also help - try naming 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste. Regular exercise and limiting caffeine can also help manage anxiety.",
    videos: [
      {
        title: "Box Breathing Animation",
        url: "https://www.youtube.com/embed/tEmt1Znux58",
        thumbnail: "/placeholder.svg?height=120&width=200&text=Box+Breathing",
      },
      {
        title: "Guided Meditation for Anxiety",
        url: "https://www.youtube.com/embed/O-6f5wQXSu8",
        thumbnail: "/placeholder.svg?height=120&width=200&text=Guided+Meditation",
      },
    ],
  },
  angry: {
    mood: "angry",
    keywords: [
      "angry",
      "mad",
      "frustrated",
      "annoyed",
      "irritated",
      "furious",
      "enraged",
      "outraged",
      "irate",
      "livid",
      "infuriated",
      "exasperated",
      "indignant",
      "resentful",
      "hostile",
      "bitter",
      "burnt out",
      "exhausted",
      "fed up",
      "aggravated",
    ],
    response:
      "I can see you're feeling frustrated right now. 😤 Your feelings are valid, and it's important to acknowledge them. Let's find a constructive way to channel this energy.",
    color: "bg-red-100 border-red-200",
    emoji: "😡",
    condition: "Anger response or possible burnout",
    remedy:
      "When you feel anger rising, try counting to 10 before responding. Physical activity can help release tension - even a short walk can make a difference. If you're experiencing burnout, consider what boundaries you might need to set and prioritize activities that replenish your energy. Writing down your thoughts can also help process these emotions.",
    videos: [
      {
        title: "Calm Down Techniques",
        url: "https://www.youtube.com/embed/BsVq5R_F6RA",
        thumbnail: "/placeholder.svg?height=120&width=200&text=Calm+Down",
      },
      {
        title: "Anger Management Strategies",
        url: "https://www.youtube.com/embed/aNHXHoV7zKQ",
        thumbnail: "/placeholder.svg?height=120&width=200&text=Anger+Management",
      },
    ],
  },
}

export function MoodAnalyzer() {
  const [input, setInput] = useState("")
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<{
    mood: string
    response: string
    color: string
    emoji: string
    condition: string
    remedy: string
    videos: Array<{ title: string; url: string; thumbnail: string }>
  } | null>(null)
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null)
  const [helpfulFeedback, setHelpfulFeedback] = useState<boolean | null>(null)

  const analyzeMood = () => {
    if (!input.trim()) return

    setAnalyzing(true)

    // Simulate processing delay
    setTimeout(() => {
      const lowerInput = input.toLowerCase()

      // Find matching mood based on keywords
      let matchedMood = null
      let highestMatchCount = 0

      for (const [mood, data] of Object.entries(moodMap)) {
        const matchCount = data.keywords.filter((keyword) => lowerInput.includes(keyword)).length

        if (matchCount > highestMatchCount) {
          highestMatchCount = matchCount
          matchedMood = mood
        }
      }

      // Default to "neutral" if no matches or very weak matches
      if (!matchedMood || highestMatchCount === 0) {
        // For simplicity, we'll default to "happy" in this demo
        // In a real app, you might have a "neutral" category
        matchedMood = "happy"
      }

      setResult(moodMap[matchedMood as keyof typeof moodMap])
      setAnalyzing(false)
      setHelpfulFeedback(null)
    }, 1500)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      analyzeMood()
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>How are you feeling?</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Describe your current mood or feelings... (e.g., 'I'm feeling a bit down today' or 'I'm so excited about my new job!')"
            className="min-h-[100px]"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <Button
            onClick={analyzeMood}
            disabled={analyzing || !input.trim()}
            className="w-full bg-gradient-to-r from-purple-500 to-pink-500"
          >
            {analyzing ? "Analyzing..." : "Analyze My Mood"}
            {!analyzing && <Send className="ml-2 h-4 w-4" />}
          </Button>
        </CardContent>
      </Card>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card className={`overflow-hidden ${result.color}`}>
              <CardContent className="p-5">
                <div className="flex items-start gap-3 mb-4">
                  <div className="text-4xl">{result.emoji}</div>
                  <div>
                    <h3 className="text-lg font-medium mb-2">{result.response}</h3>
                    <div className="text-sm text-gray-700 space-y-2">
                      <p>
                        <strong>Possible condition:</strong> {result.condition}
                      </p>
                      <p>
                        <strong>Suggested approach:</strong> {result.remedy}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Feedback section */}
                <div className="flex justify-center my-4">
                  <div className="flex items-center gap-4">
                    <p className="text-sm text-gray-600">Was this helpful?</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className={`${helpfulFeedback === true ? "bg-green-100 border-green-300" : ""}`}
                      onClick={() => setHelpfulFeedback(true)}
                    >
                      <ThumbsUp className="h-4 w-4 mr-1" />
                      Yes
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className={`${helpfulFeedback === false ? "bg-red-100 border-red-300" : ""}`}
                      onClick={() => setHelpfulFeedback(false)}
                    >
                      <ThumbsDown className="h-4 w-4 mr-1" />
                      No
                    </Button>
                  </div>
                </div>

                {/* Videos section */}
                <div className="mt-4">
                  <h4 className="font-medium mb-3">Recommended videos:</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {result.videos.map((video, index) => (
                      <Card key={index} className="overflow-hidden">
                        <div
                          className="aspect-video bg-gray-100 cursor-pointer relative"
                          onClick={() => setSelectedVideo(video.url)}
                        >
                          <img
                            src={video.thumbnail || "/placeholder.svg"}
                            alt={video.title}
                            className="w-full h-full object-cover"
                          />
                          <div className="absolute inset-0 flex items-center justify-center bg-black/20 hover:bg-black/30 transition-colors">
                            <div className="w-12 h-12 rounded-full bg-white/80 flex items-center justify-center">
                              <div className="w-0 h-0 border-t-[8px] border-t-transparent border-l-[16px] border-l-purple-600 border-b-[8px] border-b-transparent ml-1"></div>
                            </div>
                          </div>
                        </div>
                        <CardContent className="p-3">
                          <h5 className="font-medium text-sm">{video.title}</h5>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>

                {/* Video player modal */}
                {selectedVideo && (
                  <div
                    className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
                    onClick={() => setSelectedVideo(null)}
                  >
                    <div className="relative w-full max-w-3xl" onClick={(e) => e.stopPropagation()}>
                      <Button
                        className="absolute -top-10 right-0 bg-white text-black hover:bg-gray-200"
                        onClick={() => setSelectedVideo(null)}
                      >
                        Close
                      </Button>
                      <div className="aspect-video w-full">
                        <iframe
                          src={selectedVideo}
                          className="w-full h-full"
                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                          allowFullScreen
                          title="Embedded YouTube Video"
                        ></iframe>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
