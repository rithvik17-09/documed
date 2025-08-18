"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Play } from "lucide-react"

type Guide = {
  id: string
  title: string
  description: string
  videoUrl: string
}

const guides: Guide[] = [
  {
    id: "cpr",
    title: "CPR Basics",
    description: "Learn the essential steps of cardiopulmonary resuscitation (CPR)",
    videoUrl: "/videos/cpr.mp4", // put video inside public/videos
  },
  {
    id: "burns",
    title: "Treating Burns",
    description: "First aid steps for different types of burns",
    videoUrl: "/videos/burns.mp4",
  },
]

export function FirstAidGuides() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {guides.map((guide) => (
        <Card key={guide.id} className="overflow-hidden">
          <CardContent className="p-4">
            <h3 className="text-lg font-semibold">{guide.title}</h3>
            <p className="text-sm text-gray-600">{guide.description}</p>

            <div className="mt-3 flex gap-2">
              {/* Video Modal */}
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="outline" className="flex items-center gap-2">
                    <Play size={16} /> Watch Video
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[600px]">
                  <DialogHeader>
                    <DialogTitle>{guide.title}</DialogTitle>
                  </DialogHeader>
                  <video src={guide.videoUrl} controls autoPlay className="w-full rounded-lg" />
                </DialogContent>
              </Dialog>

              <Button className="bg-blue-500 hover:bg-blue-600">View Steps</Button>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
