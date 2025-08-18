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

// Keyword-based responses for first aid
const firstAidResponses = {
  burn: "For minor burns: 1) Cool the burn with cool (not cold) running water for 10-15 minutes. 2) Remove jewelry and tight items from the burned area. 3) Apply aloe vera or moisturizer. 4) Bandage loosely with sterile gauze. 5) Take an over-the-counter pain reliever if needed. Seek medical attention for severe burns, burns on the face/hands/feet, or if the burn is larger than 3 inches.",

  cpr: "CPR steps: 1) Check if the person is responsive. 2) Call emergency services (911). 3) Place the person on their back on a firm surface. 4) Place your hands on the center of their chest. 5) Perform chest compressions at a rate of 100-120 per minute, pushing at least 2 inches deep. 6) If trained, give rescue breaths after 30 compressions. 7) Continue until help arrives. Remember, proper training is recommended for performing CPR effectively.",

  choking:
    "For choking: 1) Ask 'Are you choking?' If they can't speak, cough, or breathe, act quickly. 2) Stand behind them and wrap your arms around their waist. 3) Make a fist with one hand and place it above their navel. 4) Grab your fist with your other hand and pull inward and upward with quick thrusts. 5) Repeat until the object is expelled or the person becomes unconscious. If unconscious, begin CPR.",

  bleeding:
    "To control bleeding: 1) Apply direct pressure on the wound with a clean cloth or bandage. 2) If blood soaks through, add more material without removing the first layer. 3) If possible, elevate the wounded area above the heart. 4) If bleeding is severe and doesn't stop with pressure, apply pressure to the appropriate artery (brachial for arm, femoral for leg). 5) As a last resort for life-threatening limb bleeding, apply a tourniquet if you have proper training. 6) Seek immediate medical attention.",

  "heart attack":
    "Heart attack signs include chest pain/pressure, pain radiating to arm/jaw/back, shortness of breath, cold sweat, nausea, and lightheadedness. If you suspect a heart attack: 1) Call emergency services (911) immediately. 2) Have the person sit down and rest. 3) Loosen tight clothing. 4) If the person is not allergic to aspirin and has no bleeding disorders, they may chew an aspirin (if advised by emergency services). 5) If the person becomes unconscious, begin CPR if trained.",

  stroke:
    "Remember FAST for stroke symptoms: Face drooping, Arm weakness, Speech difficulty, Time to call 911. Additional symptoms include sudden numbness, confusion, trouble seeing, dizziness, or severe headache. If you suspect a stroke: 1) Note the time symptoms started. 2) Call emergency services (911) immediately. 3) Have the person lie down with their head slightly elevated. 4) Do not give them anything to eat or drink. 5) If unconscious but breathing, place them in the recovery position.",

  fracture:
    "For a suspected fracture: 1) Keep the injured area immobile and supported. 2) Apply ice wrapped in a cloth to reduce swelling (20 minutes on, 20 minutes off). 3) If possible, elevate the injured area. 4) For an open fracture (bone piercing skin), cover the wound with a clean cloth but don't push the bone back in. 5) Do not attempt to straighten a deformed limb. 6) Seek medical attention promptly.",

  sprain:
    "For sprains, remember RICE: Rest the injured area, Ice for 20 minutes every 2-3 hours, Compress with an elastic bandage, and Elevate above heart level when possible. Take over-the-counter pain relievers if needed. If you can't bear weight on the injury, there's significant swelling or bruising, or you heard a pop when injured, seek medical attention as it could be a fracture.",

  "allergic reaction":
    "For mild allergic reactions, antihistamines may help. For severe reactions (anaphylaxis) with symptoms like difficulty breathing, swelling of face/throat, rapid pulse, or dizziness: 1) Call emergency services (911). 2) If the person has an epinephrine auto-injector (EpiPen), help them use it. 3) Have them lie still with legs elevated. 4) If unconscious but breathing, place in recovery position. 5) If not breathing, begin CPR if trained.",

  poisoning:
    "For poisoning: 1) Call poison control (1-800-222-1222) or emergency services immediately. 2) Do not induce vomiting unless specifically instructed by professionals. 3) If the poison is on the skin, remove contaminated clothing and rinse skin with running water for 15-20 minutes. 4) If the poison is inhaled, get the person to fresh air. 5) Bring the poison container or substance to the hospital if possible.",

  seizure:
    "For seizures: 1) Gently guide the person to the floor and clear the area of anything hazardous. 2) Place something soft under their head. 3) Turn them onto their side to prevent choking. 4) Do NOT restrain them or put anything in their mouth. 5) Time the seizure. 6) Stay with them until they are fully conscious. 7) Call emergency services if the seizure lasts longer than 5 minutes, if the person doesn't wake up, if another seizure occurs, or if the person is pregnant or has diabetes.",

  "heat stroke":
    "Heat stroke is a medical emergency. Symptoms include high body temperature, altered mental state, hot/dry skin, rapid breathing, racing heart, and headache. 1) Call emergency services (911). 2) Move the person to a cool place. 3) Remove excess clothing. 4) Cool the person with whatever means available—cold water, ice packs to groin/armpits/neck, or a cool bath. 5) Do not give the person anything to drink if they're not fully alert.",

  hypothermia:
    "For hypothermia: 1) Call emergency services (911). 2) Move the person to a warm place and remove wet clothing. 3) Warm the person's core first (chest, neck, head) with dry blankets or body heat. 4) If the person is alert, give warm (not hot) beverages. 5) Do not apply direct heat like hot water or heating pads, which can damage the skin or cause irregular heartbeats. 6) Handle the person gently and do not massage their limbs.",
}

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

  const getFirstAidResponse = (input: string) => {
    const lowerInput = input.toLowerCase()

    // Check for matches in our first aid responses
    for (const [keyword, response] of Object.entries(firstAidResponses)) {
      if (lowerInput.includes(keyword)) {
        return response
      }
    }

    // Default response if no matches
    return "I don't have specific information about that situation. For any medical emergency, it's best to call emergency services (911) immediately. For non-emergencies, consult with a healthcare professional for proper advice."
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
      const responseText = getFirstAidResponse(userMessage.content)

      // Add disclaimer
      const fullResponse = `${responseText}\n\nIMPORTANT: This is general first aid guidance only. Always call emergency services (911) for serious situations. This information is not a substitute for professional medical advice or training.`

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
modify in this and tell me how to add 
