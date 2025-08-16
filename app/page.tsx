"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import {
  Search,
  MessageCircle,
  BookOpen,
  Moon,
  Sun,
  Heart,
  Smile,
  Activity,
  FileText,
  Upload,
  X,
  AlertTriangle,
  CheckCircle,
  Scan,
} from "lucide-react"

export default function DocuMedApp() {
  const [isDark, setIsDark] = useState(false)
  const [activeTab, setActiveTab] = useState("DocMate")
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const toggleTheme = () => {
    setIsDark(!isDark)
    document.documentElement.classList.toggle("dark")
  }

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file)
      const reader = new FileReader()
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string)
      }
      reader.readAsDataURL(file)
      setAnalysisResult(null)
    }
  }

  const analyzeXray = async () => {
    if (!selectedImage) return

    setIsAnalyzing(true)
    const formData = new FormData()
    formData.append("image", selectedImage)

    try {
      const response = await fetch("/api/analyze-xray", {
        method: "POST",
        body: formData,
      })

      if (response.ok) {
        const result = await response.json()
        setAnalysisResult(result.analysis)
      } else {
        setAnalysisResult("Error analyzing X-ray. Please try again.")
      }
    } catch (error) {
      console.error("Analysis error:", error)
      setAnalysisResult("Error analyzing X-ray. Please try again.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const clearImage = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setAnalysisResult(null)
  }

  const firstAidVideos = [
    {
      id: 1,
      title: "CPR Emergency Response",
      description: "Learn life-saving CPR techniques for adults",
      thumbnail: "/cpr-training-emergency.png",
      videoUrl: "https://www.youtube.com/embed/TlK2gLZTsQs",
      duration: "5:32",
    },
    {
      id: 2,
      title: "Choking First Aid",
      description: "How to help someone who is choking",
      thumbnail: "/heimlich-maneuver.png",
      videoUrl: "https://www.youtube.com/embed/7CgtIgSyAiU",
      duration: "3:45",
    },
  ]

  const renderContent = () => {
    switch (activeTab) {
      case "DocMate":
        return (
          <>
            {/* DocMate Section */}
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-blue-600 mb-1">DocMate</h2>
              <p className="text-muted-foreground text-sm">Your first-aid assistant</p>
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <Button variant="outline" className="h-12 flex items-center gap-2 bg-transparent">
                <MessageCircle className="h-4 w-4" />
                Chat Assistant
              </Button>
              <Button
                variant="outline"
                className="h-12 flex items-center gap-2 bg-blue-50 border-blue-200 text-blue-700"
              >
                <BookOpen className="h-4 w-4" />
                First Aid Guides
              </Button>
            </div>

            {/* Search Bar */}
            <div className="relative mb-6">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search first aid guides..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Video Sections */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium mb-4">Essential First Aid Videos</h3>

              {firstAidVideos.map((video) => (
                <Card key={video.id} className="overflow-hidden">
                  <CardContent className="p-0">
                    <div className="aspect-video relative">
                      <iframe
                        src={video.videoUrl}
                        title={video.title}
                        className="w-full h-full"
                        frameBorder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                      />
                    </div>
                    <div className="p-4">
                      <h4 className="font-semibold text-lg mb-2">{video.title}</h4>
                      <p className="text-muted-foreground text-sm mb-2">{video.description}</p>
                      <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">
                        Duration: {video.duration}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Additional Resources */}
            <div className="mt-8">
              <h3 className="text-lg font-medium mb-4">Quick Reference Guides</h3>
              <div className="grid grid-cols-2 gap-3">
                <Button variant="outline" className="h-16 flex flex-col items-center gap-1 bg-transparent">
                  <Heart className="h-5 w-5 text-red-500" />
                  <span className="text-xs">Heart Attack</span>
                </Button>
                <Button variant="outline" className="h-16 flex flex-col items-center gap-1 bg-transparent">
                  <Activity className="h-5 w-5 text-orange-500" />
                  <span className="text-xs">Stroke Signs</span>
                </Button>
                <Button variant="outline" className="h-16 flex flex-col items-center gap-1 bg-transparent">
                  <FileText className="h-5 w-5 text-blue-500" />
                  <span className="text-xs">Wound Care</span>
                </Button>
                <Button variant="outline" className="h-16 flex flex-col items-center gap-1 bg-transparent">
                  <Smile className="h-5 w-5 text-green-500" />
                  <span className="text-xs">Allergic Reaction</span>
                </Button>
              </div>
            </div>
          </>
        )

      case "X-Ray":
        return (
          <>
            {/* X-Ray Analysis Section */}
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-purple-600 mb-1">X-Ray Analysis</h2>
              <p className="text-muted-foreground text-sm">AI-powered medical imaging analysis</p>
            </div>

            <Card className="p-4">
              <CardContent className="p-0">
                {!imagePreview ? (
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                    <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-muted-foreground mb-4">Upload an X-ray image for AI analysis</p>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                      id="xray-upload"
                    />
                    <Button asChild variant="outline">
                      <label htmlFor="xray-upload" className="cursor-pointer">
                        Choose X-ray Image
                      </label>
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="relative">
                      <img
                        src={imagePreview || "/placeholder.svg"}
                        alt="X-ray preview"
                        className="w-full max-h-64 object-contain rounded-lg border"
                      />
                      <Button variant="destructive" size="icon" className="absolute top-2 right-2" onClick={clearImage}>
                        <X className="h-4 w-4" />
                      </Button>
                    </div>

                    <div className="flex gap-2">
                      <Button onClick={analyzeXray} disabled={isAnalyzing} className="flex-1">
                        {isAnalyzing ? "Analyzing..." : "Analyze X-ray"}
                      </Button>
                    </div>

                    {analysisResult && (
                      <Card className="mt-4">
                        <CardContent className="p-4">
                          <div className="flex items-start gap-3">
                            {analysisResult.toLowerCase().includes("error") ? (
                              <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
                            ) : (
                              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                            )}
                            <div>
                              <h4 className="font-semibold mb-2">Analysis Result</h4>
                              <p className="text-sm text-muted-foreground whitespace-pre-wrap">{analysisResult}</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* X-Ray Information */}
            <div className="mt-8">
              <h3 className="text-lg font-medium mb-4">About X-Ray Analysis</h3>
              <Card className="p-4">
                <CardContent className="p-0">
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p>• AI analysis can help identify potential abnormalities in X-ray images</p>
                    <p>• Results are for informational purposes only</p>
                    <p>• Always consult with a qualified medical professional</p>
                    <p>• Supported formats: JPEG, PNG, WebP</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </>
        )

      case "MediMate":
        return (
          <div className="text-center py-20">
            <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-semibold mb-2">MediMate</h2>
            <p className="text-muted-foreground">Medication management coming soon</p>
          </div>
        )

      case "MediMood":
        return (
          <div className="text-center py-20">
            <Smile className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-semibold mb-2">MediMood</h2>
            <p className="text-muted-foreground">Mood tracking coming soon</p>
          </div>
        )

      case "PulseChain":
        return (
          <div className="text-center py-20">
            <Activity className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-semibold mb-2">PulseChain</h2>
            <p className="text-muted-foreground">Vital signs monitoring coming soon</p>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className={`min-h-screen ${isDark ? "dark" : ""}`}>
      <div className="bg-background text-foreground min-h-screen">
        {/* Header */}
        <header className="flex items-center justify-between p-4 border-b">
          <h1 className="text-2xl font-bold text-teal-600">Documed</h1>
          <Button variant="ghost" size="icon" onClick={toggleTheme} className="rounded-full">
            {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </Button>
        </header>

        {/* Main Content */}
        <main className="p-4 pb-20">{renderContent()}</main>

        {/* Bottom Navigation */}
        <nav className="fixed bottom-0 left-0 right-0 bg-background border-t">
          <div className="grid grid-cols-5 h-16">
            {[
              { name: "MediMate", icon: FileText },
              { name: "MediMood", icon: Smile },
              { name: "PulseChain", icon: Activity },
              { name: "DocMate", icon: Heart },
              { name: "X-Ray", icon: Scan },
            ].map((tab) => (
              <button
                key={tab.name}
                onClick={() => setActiveTab(tab.name)}
                className={`flex flex-col items-center justify-center gap-1 ${
                  activeTab === tab.name ? "text-blue-600 bg-blue-50" : "text-muted-foreground"
                }`}
              >
                <tab.icon className="h-5 w-5" />
                <span className="text-xs">{tab.name}</span>
              </button>
            ))}
          </div>
        </nav>
      </div>
    </div>
  )
}
