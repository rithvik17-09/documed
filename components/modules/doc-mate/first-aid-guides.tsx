"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { PlayCircle, FileText } from "lucide-react"
import { motion } from "framer-motion"

type Guide = {
  id: string
  title: string
  description: string
  videoUrl: string
  steps: string[]
}

const guides: Guide[] = [
  {
    id: "cpr",
    title: "CPR Basics",
    description: "Learn how to perform cardiopulmonary resuscitation effectively.",
    videoUrl: "/videos/cpr.mp4",
    steps: [
      "Check if the person is responsive.",
      "Call emergency services (911).",
      "Place hands on the chest and push hard and fast.",
      "Give 30 compressions followed by 2 rescue breaths.",
      "Continue until help arrives."
    ],
  },
  {
    id: "burns",
    title: "Treating Burns",
    description: "Immediate steps to take when someone suffers a burn.",
    videoUrl: "/videos/burns.mp4",
    steps: [
      "Cool the burn with running water for 10-15 minutes.",
      "Remove any tight items from the burned area.",
      "Cover with a sterile, non-fluffy bandage.",
      "Do not apply oils or butter.",
      "Seek medical help for severe burns."
    ],
  },
]

export function FirstAidGuides() {
  const [selectedGuide, setSelectedGuide] = useState<Guide | null>(null)
  const [videoOpen, setVideoOpen] = useState(false)
  const [stepsOpen, setStepsOpen] = useState(false)

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {guides.map((guide, index) => (
        <motion.div
          key={guide.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <Card className="bg-gradient-to-br from-white to-blue-50 border-blue-100 shadow-md">
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold text-blue-700">{guide.title}</h3>
              <p className="text-gray-600 mb-3">{guide.description}</p>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => {
                    setSelectedGuide(guide)
                    setVideoOpen(true)
                  }}
                >
                  <PlayCircle className="h-4 w-4" />
                  Watch Video
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => {
                    setSelectedGuide(guide)
                    setStepsOpen(true)
                  }}
                >
                  <FileText className="h-4 w-4" />
                  View Steps
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      ))}

      {/* Video Dialog */}
      <Dialog open={videoOpen} onOpenChange={setVideoOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{selectedGuide?.title} - Video</DialogTitle>
          </DialogHeader>
          {selectedGuide && (
            <video
              src={selectedGuide.videoUrl}
              controls
              className="w-full rounded-lg"
            />
          )}
        </DialogContent>
      </Dialog>

      {/* Steps Dialog */}
      <Dialog open={stepsOpen} onOpenChange={setStepsOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{selectedGuide?.title} - Steps</DialogTitle>
          </DialogHeader>
          <ol className="list-decimal pl-5 space-y-2">
            {selectedGuide?.steps.map((step, idx) => (
              <li key={idx} className="text-gray-700">
                {step}
              </li>
            ))}
          </ol>
        </DialogContent>
      </Dialog>
    </div>
  )
}
