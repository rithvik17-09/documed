"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

export default function Home() {
  const [activeTab, setActiveTab] = useState("DocMate")

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navbar */}
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-blue-600">DocuMed</h1>
          <nav className="space-x-4">
            {["DocMate", "Prescription", "First Aid", "Research", "X-Ray"].map((tab) => (
              <Button
                key={tab}
                variant={activeTab === tab ? "default" : "ghost"}
                onClick={() => setActiveTab(tab)}
              >
                {tab}
              </Button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 container mx-auto px-6 py-8">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === "DocMate" && <DocMate />}
          {activeTab === "Prescription" && <Prescription />}
          {activeTab === "First Aid" && <FirstAid />}
          {activeTab === "Research" && <Research />}
          {activeTab === "X-Ray" && <XRayAnalysis />}
        </motion.div>
      </main>
    </div>
  )
}

/* Components for each section */
function DocMate() {
  return (
    <Card>
      <CardContent className="p-6">
        <h2 className="text-xl font-semibold mb-4">DocMate</h2>
        <p className="text-gray-600">AI-powered chatbot for medical guidance.</p>
      </CardContent>
    </Card>
  )
}

function Prescription() {
  return (
    <Card>
      <CardContent className="p-6 space-y-4">
        <h2 className="text-xl font-semibold">Prescription Generator</h2>
        <Input placeholder="Enter symptoms" />
        <Button>Generate Prescription</Button>
      </CardContent>
    </Card>
  )
}

function FirstAid() {
  return (
    <Card>
      <CardContent className="p-6">
        <h2 className="text-xl font-semibold">First Aid Assistant</h2>
        <Textarea placeholder="Enter emergency situation..." />
        <Button className="mt-2">Get Instructions</Button>
      </CardContent>
    </Card>
  )
}

function Research() {
  return (
    <Card>
      <CardContent className="p-6">
        <h2 className="text-xl font-semibold">Research Assistant</h2>
        <Input placeholder="Enter research topic" />
        <Button className="mt-2">Find Papers</Button>
      </CardContent>
    </Card>
  )
}

function XRayAnalysis() {
  return (
    <Card>
      <CardContent className="p-6">
        <h2 className="text-xl font-semibold">X-Ray Analysis</h2>
        <Input type="file" accept="image/*" />
        <Button className="mt-2">Analyze</Button>
      </CardContent>
    </Card>
  )
}
