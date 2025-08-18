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
        url: "https://www.youtube.com/watch?v=inpok4MKVLM",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Joyful+Meditation`,
      },
      {
        type: "video" as const,
        title: "Happiness Habits",
        description: "Daily practices to maintain your positive mood",
        url: "https://www.youtube.com/watch?v=92i5m3tV5XY",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Happiness+Habits`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Upbeat Playlist",
        description: "Energetic songs to keep your spirits high",
        url: "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Upbeat+Music`,
      },
      {
        type: "music" as const,
        title: "Happy Classics",
        description: "Timeless songs that bring joy",
        url: "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Happy+Classics`,
      },
    ],
    reading: [
      {
        type: "reading" as const,
        title: "The Science of Happiness",
        description: "Research-backed ways to maintain happiness",
        url: "https://greatergood.berkeley.edu/images/tools/ScienceofHappiness.pdf",
      },
      {
        type: "reading" as const,
        title: "Positive Affirmations",
        description: "Daily affirmations to reinforce your positive mindset",
        url: "https://www.louisehay.com/affirmations/",
      },
    ],
  },
  Calm: {
    videos: [
      {
        type: "video" as const,
        title: "Peaceful Meditation",
        description: "A guided meditation to maintain your calm state",
        url: "https://www.youtube.com/watch?v=MIr3RsUWrdo",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Peaceful+Meditation`,
      },
      {
        type: "video" as const,
        title: "Mindfulness Practices",
        description: "Simple techniques to stay present and calm",
        url: "https://www.youtube.com/watch?v=ZToicYcHIOU",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Mindfulness+Practices`,
      },
    ],
    music: [
      {
        type: "music" as const,
        title: "Ambient Sounds",
        description: "Gentle environmental sounds for relaxation",
        url: "https://www.youtube.com/watch?v=1ZYbU82GVz4",
        thumbnail: `/placeholder.svg?height=120&width=200&text=Ambient+Sounds`,
      },
      {
        type: "music" as const,
        title: "Calming Classical",
        description: "Soothing classical pieces for tranquility",
        url: "https://open.spotify.com/playlist/37i9dQZF1DWWEJlAGA9gs0",
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
    videos: [],
    music: [],
    reading: [],
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

        {/* Videos */}
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
                  <a href={video.url} target="_blank" rel="noopener noreferrer">
                    <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">Watch Now</Button>
                  </a>
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

        {/* Music */}
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
                  <a href={item.url} target="_blank" rel="noopener noreferrer">
                    <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">Listen Now</Button>
                  </a>
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

        {/* Reading */}
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
                  <a href={item.url} target="_blank" rel="noopener noreferrer">
                    <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">Read Now</Button>
                  </a>
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
