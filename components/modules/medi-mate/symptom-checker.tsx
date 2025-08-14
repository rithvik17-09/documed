"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Loader2, Send, Plus } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { motion } from "framer-motion"

type Message = {
  role: "user" | "assistant"
  content: string
}

// Keyword-based responses for symptom checker
const symptomResponses = {
  headache:
    "Headaches can be caused by stress, dehydration, or eye strain. Try to rest, drink water, and reduce screen time. If it persists or is severe, consult a doctor.",
  fever:
    "Fever is often a sign that your body is fighting an infection. Rest, stay hydrated, and take acetaminophen if needed. If fever is high (above 103°F/39.4°C) or lasts more than 3 days, seek medical attention.",
  cough:
    "For a dry cough, try honey in warm water or tea. For a productive cough, stay hydrated and consider an expectorant. If you have difficulty breathing or the cough persists for more than 2 weeks, consult a doctor.",
  "sore throat":
    "Gargle with warm salt water, drink warm liquids, and use throat lozenges. If you have difficulty swallowing or the pain is severe, see a healthcare provider.",
  fatigue:
    "Fatigue can be caused by lack of sleep, poor diet, stress, or underlying medical conditions. Ensure you're getting enough rest, eating well, and staying hydrated. If fatigue persists, consult a doctor.",
  nausea:
    "Try eating small, bland meals and staying hydrated. Avoid strong odors and greasy foods. If vomiting occurs or nausea persists for more than 2 days, seek medical attention.",
  dizziness:
    "Dizziness can be caused by dehydration, inner ear issues, or low blood sugar. Sit or lie down, drink water, and have a small snack. If dizziness is severe or accompanied by other symptoms, seek medical help.",
  "chest pain":
    "Chest pain could be serious and might indicate a heart problem. If you're experiencing chest pain, especially if it's severe or accompanied by shortness of breath, seek emergency medical attention immediately.",
  "abdominal pain":
    "Try resting, applying a heating pad, and avoiding foods that might irritate your stomach. If the pain is severe, persistent, or accompanied by fever, seek medical attention.",
  rash: "Avoid scratching and use mild, fragrance-free soap. Apply a cold compress or hydrocortisone cream. If the rash is widespread, painful, or accompanied by other symptoms, consult a doctor.",
  "shortness of breath":
    "This could be serious. Sit upright, try to stay calm, and use any prescribed inhalers if you have them. If the symptom is severe or new, seek emergency medical attention.",
  "back pain":
    "Rest, apply ice for the first 48-72 hours, then switch to heat. Over-the-counter pain relievers may help. If pain is severe or radiates down your legs, consult a healthcare provider.",
  "joint pain":
    "Rest the affected joint, apply ice to reduce swelling, and consider over-the-counter pain relievers. If pain is severe or accompanied by significant swelling or redness, see a doctor.",
  diarrhea:
    "Stay hydrated with water, clear broths, and electrolyte solutions. Eat bland foods like bananas, rice, and toast. If diarrhea is severe, contains blood, or lasts more than 2 days, seek medical attention.",
  vomiting:
    "Sip clear liquids slowly and avoid solid foods until vomiting subsides. If vomiting is severe, contains blood, or lasts more than 24 hours, seek medical help.",
  "muscle aches":
    "Rest, apply ice to reduce inflammation, and consider over-the-counter pain relievers. If muscle pain is severe or accompanied by redness or swelling, consult a doctor.",
  chills:
    "Chills often accompany fever. Stay warm, rest, and drink plenty of fluids. If chills are severe or persistent, seek medical attention.",
  "loss of appetite":
    "Try eating small, frequent meals of foods you enjoy. If loss of appetite is accompanied by weight loss or lasts more than a few days, consult a healthcare provider.",
}

export function SymptomChecker() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm your symptom checker assistant. Please describe your symptoms, and I'll try to help you understand what might be going on.",
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [commonSymptoms, setCommonSymptoms] = useState([
    "Headache",
    "Fever",
    "Cough",
    "Sore throat",
    "Fatigue",
    "Nausea",
  ])

  const getSymptomResponse = (input: string) => {
    const lowerInput = input.toLowerCase()

    // Check for matches in our symptom responses
    for (const [symptom, response] of Object.entries(symptomResponses)) {
      if (lowerInput.includes(symptom)) {
        return response
      }
    }

    // Default response if no matches
    return "I'm not familiar with those specific symptoms. It's best to consult with a healthcare professional for proper evaluation and advice."
  }

  const handleSend = async () => {
    if (!input.trim()) return

    // Add user message
    const userMessage: Message = { role: "user", content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    // Simulate API delay
    setTimeout(() => {
      // Generate response based on keywords
      const responseText = getSymptomResponse(userMessage.content)

      // Add disclaimer
      const fullResponse = `${responseText}\n\nRemember: This is general information only and not a substitute for professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.`

      // Add assistant message
      setMessages((prev) => [...prev, { role: "assistant", content: fullResponse }])
      setIsLoading(false)
    }, 1000)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const addSymptom = (symptom: string) => {
    setInput((prev) => (prev ? `${prev}, ${symptom.toLowerCase()}` : symptom.toLowerCase()))
  }

  return (
    <div className="flex flex-col h-[calc(100vh-16rem)]">
      <Card className="flex-1 overflow-hidden bg-gradient-to-br from-teal-50 to-blue-50 border-teal-100">
        <CardContent className="p-4 h-full flex flex-col">
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
                    message.role === "user" ? "bg-teal-500 text-white" : "bg-white border border-teal-100"
                  }`}
                >
                  {message.content}
                </div>
              </motion.div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-[80%] rounded-lg p-3 bg-white border border-teal-100">
                  <Loader2 className="h-5 w-5 animate-spin text-teal-500" />
                </div>
              </div>
            )}
          </div>

          <div className="pt-2 border-t border-teal-100">
            <div className="flex gap-2 mb-2 overflow-x-auto pb-2">
              {commonSymptoms.map((symptom) => (
                <Button
                  key={symptom}
                  variant="outline"
                  size="sm"
                  className="whitespace-nowrap bg-white"
                  onClick={() => addSymptom(symptom)}
                >
                  {symptom}
                </Button>
              ))}
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm" className="whitespace-nowrap bg-white">
                    <Plus className="h-4 w-4 mr-1" /> More
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Common Symptoms</DialogTitle>
                  </DialogHeader>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      "Abdominal pain",
                      "Chest pain",
                      "Dizziness",
                      "Shortness of breath",
                      "Rash",
                      "Joint pain",
                      "Back pain",
                      "Diarrhea",
                      "Vomiting",
                      "Muscle aches",
                      "Chills",
                      "Loss of appetite",
                    ].map((symptom) => (
                      <Button
                        key={symptom}
                        variant="outline"
                        className="justify-start"
                        onClick={() => {
                          addSymptom(symptom)
                        }}
                      >
                        {symptom}
                      </Button>
                    ))}
                  </div>
                </DialogContent>
              </Dialog>
            </div>

            <div className="flex gap-2">
              <Input
                placeholder="Describe your symptoms..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                className="bg-white"
              />
              <Button onClick={handleSend} disabled={isLoading || !input.trim()}>
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
