"use client"

import type React from "react"
import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2, Send, Video, BookOpen, Music } from "lucide-react"

type ContentItem = {
  title: string
  description: string
  url: string
}

type MoodContent = {
  reading: ContentItem[]
  videos: ContentItem[]
  music: ContentItem[]
}

export default function MoodCompanion() {
  const [mood, setMood] = useState("")
  const [content, setContent] = useState<MoodContent>({
    reading: [],
    videos: [],
    music: [],
  })
  const [isLoading, setIsLoading] = useState(false)

  const handleGenerate = async () => {
    if (!mood.trim()) return
    setIsLoading(true)

    // Simulated API response
    setTimeout(() => {
      setContent({
        reading: [
          {
            title: "The Power of Now",
            description: "A guide to spiritual enlightenment by Eckhart Tolle",
            url: "https://example.com/power-of-now",
          },
        ],
        videos: [
          {
            title: "Motivational Video",
            description: "Stay motivated with this short inspirational video",
            url: "https://youtube.com/example",
          },
        ],
        music: [
          {
            title: "Relaxing Piano Music",
            description: "Soothing instrumental music to calm your mind",
            url: "https://spotify.com/example",
          },
        ],
      })
      setIsLoading(false)
    }, 1500)
  }

  return (
    <div className="w-full max-w-3xl mx-auto p-4 space-y-6">
      {/* Input Section */}
      <Card className="bg-gradient-to-r from-purple-50 to-pink-50">
        <CardContent className="p-6 space-y-4">
          <h2 className="text-2xl font-bold text-center bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            Mood Companion
          </h2>
          <div className="flex gap-2">
            <Input
              placeholder="How are you feeling today?"
              value={mood}
              onChange={(e) => setMood(e.target.value)}
              className="flex-1"
            />
            <Button
              onClick={handleGenerate}
              disabled={isLoading}
              className="bg-gradient-to-r from-purple-500 to-pink-500 text-white"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Content Section */}
      <Tabs defaultValue="reading" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
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

        {/* Reading Section */}
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
                    <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">
                      Read Now
                    </Button>
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

        {/* Videos Section */}
        <TabsContent value="videos" className="space-y-4">
          {isLoading ? (
            <Card className="bg-gray-50">
              <CardContent className="p-8 flex justify-center items-center">
                <Loader2 className="h-8 w-8 animate-spin text-purple-500" />
              </CardContent>
            </Card>
          ) : content.videos.length > 0 ? (
            content.videos.map((item, index) => (
              <Card key={index}>
                <CardContent className="p-4">
                  <h4 className="font-medium">{item.title}</h4>
                  <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                  <a href={item.url} target="_blank" rel="noopener noreferrer">
                    <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">
                      Watch Now
                    </Button>
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

        {/* Music Section */}
        <TabsContent value="music" className="space-y-4">
          {isLoading ? (
            <Card className="bg-gray-50">
              <CardContent className="p-8 flex justify-center items-center">
                <Loader2 className="h-8 w-8 animate-spin text-purple-500" />
              </CardContent>
            </Card>
          ) : content.music.length > 0 ? (
            content.music.map((item, index) => (
              <Card key={index}>
                <CardContent className="p-4">
                  <h4 className="font-medium">{item.title}</h4>
                  <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                  <a href={item.url} target="_blank" rel="noopener noreferrer">
                    <Button className="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500">
                      Listen Now
                    </Button>
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
      </Tabs>
    </div>
  )
}
