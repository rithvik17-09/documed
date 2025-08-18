"use client"

import type React from "react"
import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2, Send, Video, BookOpen } from "lucide-react"
import { motion } from "framer-motion"
import { FirstAidGuides } from "./first-aid-guides"

type Message = {
  role: "user" | "assistant"
  content: string
}

// same firstAidResponses object as before...
// (unchanged for brevity)

export default function DocMate() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm DocMate, your first-aid assistant. How can I help you today?",
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("chat")

  // ...same functions as before (getFirstAidResponse, handleSend, handleKeyDown)...

  const suggestedQueries = [
    "What should I do for a burn?",
    "How to perform CPR?",
    "Signs of a heart attack",
    "How to treat a sprained ankle",
    "First aid for choking",
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.3 }}
      className="space-y-4"
    >
      <h2 className="text-xl font-semibold text-blue-700">DocMate</h2>
      <p className="text-gray-600">Your first-aid assistant</p>

      <Tabs defaultValue="chat" className="w-full" onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-2 mb-4">
          <TabsTrigger value="chat">
            <BookOpen className="h-4 w-4 mr-2" />
            Chat Assistant
          </TabsTrigger>
          <TabsTrigger value="guides">
            <Video className="h-4 w-4 mr-2" />
            First Aid Guides
          </TabsTrigger>
        </TabsList>

        <TabsContent value="chat" className="space-y-4">
          {/* chat UI unchanged */}
          <Card className="flex-1 overflow-hidden bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-100">
            <CardContent className="p-4 h-[calc(100vh-20rem)] flex flex-col">
              <div className="flex-1 overflow-y-auto space-y-4 pb-4">
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-3 ${
                        message.role === "user" ? "bg-blue-500 text-white" : "bg-white border border-blue-100"
                      }`}
                    >
                      {message.content}
                    </div>
                  </motion.div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] rounded-lg p-3 bg-white border border-blue-100">
                      <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                    </div>
                  </div>
                )}
              </div>

              {messages.length === 1 && (
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-gray-500 mb-2">Suggested questions:</h4>
                  <div className="flex flex-wrap gap-2">
                    {suggestedQueries.map((query) => (
                      <Button
                        key={query}
                        variant="outline"
                        size="sm"
                        className="bg-white"
                        onClick={() => {
                          setInput(query)
                          setTimeout(() => handleSend(), 100)
                        }}
                      >
                        {query}
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              <div className="pt-2 border-t border-blue-100">
                <div className="flex gap-2">
                  <Input
                    placeholder="Ask about first aid or medical emergencies..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    className="bg-white"
                  />
                  <Button
                    onClick={handleSend}
                    disabled={isLoading || !input.trim()}
                    className="bg-blue-500 hover:bg-blue-600"
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="guides" className="space-y-4">
          <FirstAidGuides />
        </TabsContent>
      </Tabs>
    </motion.div>
  )
}
