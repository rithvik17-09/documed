"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2, Youtube, Music, BookOpen, RefreshCw } from "lucide-react"

type Mood = {
  emoji: string
  label: string
  color: string
  gradient: string
}

interface ContentSuggestionsProps {
  mood: Mood | null
}

type Content = {
  type: "video" | "music" | "reading"
  title: string
  description: string
  url: string
  thumbnail?: string
}

// Predefined content suggestions based on mood
const moodContent = {
  Happy: {
    videos: [
      {
        type: "video" as const,
        title: "Joyful Meditation",
        description: "A guided meditation to enhance your positive feelings",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Joyful+Meditation`,
      },
      {
        type: "video" as const,
        title: "Happiness Habits",
        description: "Daily practices to maintain your positive mood",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Happiness+Habits`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Upbeat Playlist",
        description: "Energetic songs to keep your spirits high",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Upbeat+Music`,
      },
      {
        type: "music" as const,
        title: "Happy Classics",
        description: "Timeless songs that bring joy",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Happy+Classics`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "The Science of Happiness",
        description: "Research-backed ways to maintain happiness",
        url: "#",
      },
      {
        type: "reading" as const,
        title: "Positive Affirmations",
        description: "Daily affirmations to reinforce your positive mindset",
        url: "#",
      },
    ],
  },
  Calm: {
    videos: [
      {
        type: "video" as const,
        title: "Peaceful Meditation",
        description: "A guided meditation to maintain your calm state",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Peaceful+Meditation`,
      },
      {
        type: "video" as const,
        title: "Mindfulness Practices",
        description: "Simple techniques to stay present and calm",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Mindfulness+Practices`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Ambient Sounds",
        description: "Gentle environmental sounds for relaxation",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Ambient+Sounds`,
      },
      {
        type: "music" as const,
        title: "Calming Classical",
        description: "Soothing classical pieces for tranquility",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Calming+Classical`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "Life of Pi",
        description: "Finding peace in everyday life",
        url: "https://www.ssgopalganj.in/online/E-Books/CLASS%20VIII/Life%20of%20Pi%20by%20Yann%20Martel.pdf",
      },
      {
        type: "reading" as const,
        title: "The Rules of Life",
        description: "Practices for a more centered existence",
        url: "http://albaab.free.fr/The%20Rules%20of%20Life.pdf",
      },
    ],
  },
  Neutral: {
    videos: [
      {
        type: "video" as const,
        title: "Balanced Living",
        description: "Finding equilibrium in daily life",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Balanced+Living`,
      },
      {
        type: "video" as const,
        title: "Mindfulness Basics",
        description: "Introduction to mindfulness practices",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Mindfulness+Basics`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Focused Playlist",
        description: "Music to help you concentrate",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Focused+Music`,
      },
      {
        type: "music" as const,
        title: "Ambient Work Sounds",
        description: "Background sounds for productivity",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Work+Sounds`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "Finding Your Purpose",
        description: "Discovering what drives you",
        url: "https://www.ssgopalganj.in/online/E-Books/CLASS%20VIII/Life%20of%20Pi%20by%20Yann%20Martel.pdf",
      },
      {
        type: "reading" as const,
        title: "Everyday Mindfulness",
        description: "Simple practices for daily awareness",
        url: "#",
      },
    ],
  },
  Sad: {
    videos: [
      {
        type: "video" as const,
        title: "Gentle Mood Lifting",
        description: "Compassionate practices for difficult emotions",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Mood+Lifting`,
      },
      {
        type: "video" as const,
        title: "Self-Compassion Guide",
        description: "Being kind to yourself during tough times",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Self+Compassion`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Comforting Melodies",
        description: "Gentle music to soothe difficult emotions",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Comforting+Music`,
      },
      {
        type: "music" as const,
        title: "Emotional Healing",
        description: "Songs that acknowledge and process feelings",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Healing+Music`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "Coping with Sadness",
        description: "Healthy ways to process difficult emotions",
        url: "#",
      },
      {
        type: "reading" as const,
        title: "Finding Hope",
        description: "Stories of resilience and overcoming challenges",
        url: "#",
      },
    ],
  },
  Angry: {
    videos: [
      {
        type: "video" as const,
        title: "Anger Management",
        description: "Techniques to process and release anger",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Anger+Management`,
      },
      {
        type: "video" as const,
        title: "Calming Exercises",
        description: "Physical practices to release tension",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Calming+Exercises`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Tension Release",
        description: "Music to help process and release anger",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Tension+Release`,
      },
      {
        type: "music" as const,
        title: "Calming Rhythms",
        description: "Beats to help regulate emotions",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Calming+Rhythms`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "Understanding Anger",
        description: "The psychology behind anger and how to manage it",
        url: "#",
      },
      {
        type: "reading" as const,
        title: "Healthy Expression",
        description: "Constructive ways to express difficult emotions",
        url: "#",
      },
    ],
  },
  Anxious: {
    videos: [
      {
        type: "video" as const,
        title: "Anxiety Relief",
        description: "Guided practices for reducing anxiety",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Anxiety+Relief`,
      },
      {
        type: "video" as const,
        title: "Grounding Techniques",
        description: "Simple methods to stay present when anxious",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Grounding+Techniques`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Calming Playlist",
        description: "Gentle music to ease anxiety",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Calming+Music`,
      },
      {
        type: "music" as const,
        title: "Nature Sounds",
        description: "Peaceful environmental sounds for relaxation",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Nature+Sounds`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "Managing Anxiety",
        description: "Evidence-based approaches to anxiety management",
        url: "#",
      },
      {
        type: "reading" as const,
        title: "Worry Less, Live More",
        description: "Practical strategies for reducing worry",
        url: "#",
      },
    ],
  },
  Tired: {
    videos: [
      {
        type: "video" as const,
        title: "Gentle Energy Boost",
        description: "Mild exercises to increase energy naturally",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Energy+Boost`,
      },
      {
        type: "video" as const,
        title: "Restorative Yoga",
        description: "Gentle yoga for when you're low on energy",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Restorative+Yoga`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Refreshing Sounds",
        description: "Music to gently energize your mind",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Refreshing+Music`,
      },
      {
        type: "music" as const,
        title: "Morning Playlist",
        description: "Gentle wake-up music to start your day",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Morning+Music`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "Energy Management",
        description: "Understanding and optimizing your energy levels",
        url: "#",
      },
      {
        type: "reading" as const,
        title: "Quality Rest",
        description: "Improving sleep and relaxation for better energy",
        url: "#",
      },
    ],
  },
  Grateful: {
    videos: [
      {
        type: "video" as const,
        title: "Gratitude Practices",
        description: "Exercises to enhance feelings of thankfulness",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Gratitude+Practices`,
      },
      {
        type: "video" as const,
        title: "Appreciation Meditation",
        description: "Guided meditation focused on gratitude",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Appreciation+Meditation`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Uplifting Classics",
        description: "Timeless music to enhance positive feelings",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Uplifting+Classics`,
      },
      {
        type: "music" as const,
        title: "Joyful Melodies",
        description: "Music to celebrate life's blessings",
        url: "#",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Joyful+Melodies`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "The Gratitude Effect",
        description: "How thankfulness transforms your life",
        url: "#",
      },
      {
        type: "reading" as const,
        title: "Everyday Blessings",
        description: "Finding wonder in ordinary moments",
        url: "#",
      },
    ],
  },
}

export function ContentSuggestions({ mood }: ContentSuggestionsProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("videos")
  const [content, setContent] = useState<{
    videos: Content[]
    music: Content[]
    reading: Content[]
  }>({
    videos: [],
    music: [],
    reading: [],
  })

  // Get content based on mood
  useEffect(() => {
    const fetchContent = async () => {
      setIsLoading(true)

      // Simulate API delay
      await new Promise((resolve) => setTimeout(resolve, 1000))

      const moodLabel = mood?.label || "Neutral"

      // Get predefined content for this mood, or default to Neutral
      const moodSpecificContent = moodContent[moodLabel as keyof typeof moodContent] || moodContent.Neutral

      setContent(moodSpecificContent)
      setIsLoading(false)
    }

    fetchContent()
  }, [mood])

  const refreshContent = () => {
    // Simulate refreshing content
    setIsLoading(true)
    setTimeout(() => setIsLoading(false), 1000)
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Suggested Content</h3>
        <Button variant="outline" size="sm" onClick={refreshContent} disabled={isLoading}>
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
        </Button>
      </div>

      <Tabs defaultValue="videos" onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-3 mb-4">
          <TabsTrigger value="videos">
            <Youtube className="h-4 w-4 mr-2" />
            Videos
          </TabsTrigger>
          <TabsTrigger value="music">
            <Music className="h-4 w-4 mr-2" />
            Music
          </TabsTrigger>
          <TabsTrigger value="reading">
            <BookOpen className="h-4 w-4 mr-2" />
            Reading
          </TabsTrigger>
        </TabsList>

        <TabsContent value="videos" className="space-y-4">
          {isLoading ? (
            <Card className="bg-gray-50">
              <CardContent className="p-8 flex justify-center items-center">
                <Loader2 className="h-8 w-8 animate-spin text-purple-500" />
              </CardContent>
            </Card>
          ) : content.videos.length > 0 ? (
            content.videos.map((video, index) => (
              <Card key={index} className="overflow-hidden">
                <div className="aspect-video bg-gray-100">
                  <img
                    src={video.thumbnail || "/placeholder.svg"}
                    alt={video.title}
                    className="w-full h-full object-cover"
                  />
                </div>
                <CardContent className="p-4">
                  <h4 className="font-medium">{video.title}</h4>
                  <p className="text-sm text-gray-500 mt-1">{video.description}</p>
                  <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">Watch Now</Button>
                </CardContent>
              </Card>
            ))
          ) : (
            <Card className="bg-gray-50">
              <CardContent className="p-8 text-center text-gray-500">
                No video suggestions available. Try selecting a mood first.
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="music" className="space-y-4">
          {isLoading ? (
            <Card className="bg-gray-50">
              <CardContent className="p-8 flex justify-center items-center">
                <Loader2 className="h-8 w-8 animate-spin text-purple-500" />
              </CardContent>
            </Card>
          ) : content.music.length > 0 ? (
            content.music.map((item, index) => (
              <Card key={index} className="overflow-hidden">
                <div className="h-24 bg-gray-100 flex">
                  <div className="w-24 h-24 bg-gray-200">
                    <img
                      src={item.thumbnail || "/placeholder.svg"}
                      alt={item.title}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="p-4 flex-1">
                    <h4 className="font-medium">{item.title}</h4>
                    <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                  </div>
                </div>
                <CardContent className="p-4 pt-0">
                  <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">Listen Now</Button>
                </CardContent>
              </Card>
            ))
          ) : (
            <Card className="bg-gray-50">
              <CardContent className="p-8 text-center text-gray-500">
                No music suggestions available. Try selecting a mood first.
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="reading" className="space-y-4">
          {isLoading ? (
            <Card className="bg-gray-50">
              <CardContent className="p-8 flex justify-center items-center">
                <Loader2 className="h-8 w-8 animate-spin text-purple-500" />
              </CardContent>
            </Card>
          ) : content.reading.length > 0 ? (
            content.reading.map((item, index) => (
              <Card key={index}>
                <CardContent className="p-4">
                  <h4 className="font-medium">{item.title}</h4>
                  <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                  <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">Read Now</Button>
                </CardContent>
              </Card>
            ))
          ) : (
            <Card className="bg-gray-50">
              <CardContent className="p-8 text-center text-gray-500">
                No reading suggestions available. Try selecting a mood first.
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}



