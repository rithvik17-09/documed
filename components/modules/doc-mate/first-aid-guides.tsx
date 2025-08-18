"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, Play, ArrowRight } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

type Guide = {
  id: string
  title: string
  description: string
  thumbnail: string
  videoUrl?: string
  steps?: Array<{
    title: string
    description: string
  }>
}

export function FirstAidGuides() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedGuide, setSelectedGuide] = useState<Guide | null>(null)

  const guides: Guide[] = [
    {
      id: "1",
      title: "CPR Basics",
      description: "Learn the essential steps of cardiopulmonary resuscitation (CPR)",
      thumbnail: "/placeholder.svg?height=120&width=200&text=CPR+Basics",
      videoUrl: "#",
      steps: [
        {
          title: "Check responsiveness",
          description: "Tap the person and shout 'Are you OK?'",
        },
        {
          title: "Call emergency services",
          description: "If unresponsive, call 911 or local emergency number",
        },
        {
          title: "Start chest compressions",
          description:
            "Place hands on center of chest and push hard and fast at a rate of 100-120 compressions per minute",
        },
        {
          title: "Give rescue breaths",
          description: "After 30 compressions, give 2 rescue breaths if trained",
        },
        {
          title: "Continue CPR",
          description: "Alternate 30 compressions with 2 breaths until help arrives",
        },
      ],
    },
    {
      id: "2",
      title: "Treating Burns",
      description: "First aid steps for different types of burns",
      thumbnail: "/placeholder.svg?height=120&width=200&text=Treating+Burns",
      videoUrl: "#",
      steps: [
        {
          title: "Stop the burning",
          description:
            "Remove the source of the burn and cool the area with cool (not cold) running water for 10-15 minutes",
        },
        {
          title: "Remove jewelry and clothing",
          description: "Take off jewelry and clothing from the burned area, unless it's stuck to the skin",
        },
        {
          title: "Cover the burn",
          description: "Cover with a sterile, non-stick bandage or clean cloth",
        },
        {
          title: "Take pain medication",
          description: "Over-the-counter pain relievers can help manage pain",
        },
        {
          title: "Seek medical attention",
          description:
            "For severe burns, burns on the face, hands, feet, genitals, or major joints, or burns larger than 3 inches",
        },
      ],
    },
    {
      id: "3",
      title: "Choking Response",
      description: "How to help someone who is choking",
      thumbnail: "/placeholder.svg?height=120&width=200&text=Choking+Response",
      videoUrl: "#",
      steps: [
        {
          title: "Recognize choking",
          description: "Look for universal choking sign (hands clutched to throat) or inability to talk",
        },
        {
          title: "Perform abdominal thrusts (Heimlich maneuver)",
          description:
            "Stand behind the person, wrap your arms around their waist, make a fist with one hand, position it above the navel, grasp your fist with the other hand, and press into the abdomen with quick, upward thrusts",
        },
        {
          title: "Continue until obstruction clears",
          description: "Repeat thrusts until the object is expelled or the person becomes unconscious",
        },
        {
          title: "If person becomes unconscious",
          description: "Lower them to the ground, call emergency services, and begin CPR",
        },
      ],
    },
    {
      id: "4",
      title: "Controlling Bleeding",
      description: "Steps to control severe bleeding",
      thumbnail: "/placeholder.svg?height=120&width=200&text=Controlling+Bleeding",
      videoUrl: "#",
      steps: [
        {
          title: "Apply direct pressure",
          description:
            "Use a clean cloth, gauze, or your hand to apply firm, steady pressure directly on the wound to stop the bleeding.",
        },
        {
          title: "Keep pressure constant",
          description:
            "Do not remove the cloth if it becomes soaked. Place another cloth or bandage on top and continue pressing.",
        },
        {
          title: "Elevate the injured area",
          description:
            "If possible, raise the bleeding limb above the level of the heart to help slow blood flow.",
        },
        {
          title: "Use a tourniquet if needed",
          description:
            "For severe bleeding that doesn’t stop with direct pressure, apply a tourniquet above the wound. Note the time it was applied.",
        },
        {
          title: "Seek medical help immediately",
          description:
            "Call emergency services or get professional medical care as soon as possible.",
        },
      ],
    },
  ]

  const filteredGuides = guides.filter(
    (guide) =>
      guide.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      guide.description.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
        <Input
          placeholder="Search first aid guides..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredGuides.map((guide) => (
          <Card key={guide.id} className="overflow-hidden">
            <div className="aspect-video bg-gray-100">
              <img
                src={guide.thumbnail || "/placeholder.svg"}
                alt={guide.title}
                className="w-full h-full object-cover"
              />
            </div>
            <CardContent className="p-4">
              <h3 className="font-medium">{guide.title}</h3>
              <p className="text-sm text-gray-500 mt-1">{guide.description}</p>
              <div className="flex gap-2 mt-3">
                {guide.videoUrl && (
                  <Button variant="outline" className="flex-1">
                    <Play className="h-4 w-4 mr-1" />
                    Watch Video
                  </Button>
                )}
                <Dialog>
                  <DialogTrigger asChild>
                    <Button className="flex-1 bg-blue-500 hover:bg-blue-600" onClick={() => setSelectedGuide(guide)}>
                      <ArrowRight className="h-4 w-4 mr-1" />
                      View Steps
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-h-[80vh] overflow-auto">
                    <DialogHeader>
                      <DialogTitle>{selectedGuide?.title}</DialogTitle>
                    </DialogHeader>
                    <div className="py-4">
                      {selectedGuide?.steps ? (
                        <div className="space-y-4">
                          {selectedGuide.steps.map((step, index) => (
                            <div key={index} className="flex gap-4">
                              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold">
                                {index + 1}
                              </div>
                              <div>
                                <h4 className="font-medium">{step.title}</h4>
                                <p className="text-sm text-gray-600 mt-1">{step.description}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-500">No detailed steps available for this guide.</p>
                      )}
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredGuides.length === 0 && (
        <Card className="bg-gray-50">
          <CardContent className="p-8 text-center text-gray-500">
            No guides found matching "{searchQuery}". Try a different search term.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
