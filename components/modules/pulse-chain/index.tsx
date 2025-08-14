"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { motion, AnimatePresence } from "framer-motion"
import { VitalsMonitor } from "./vitals-monitor"
import { EmergencyContacts } from "./emergency-contacts"
import { AlertsFeed } from "./alerts-feed"
import { SosButton } from "./sos-button"

export default function PulseChain() {
  const [activeTab, setActiveTab] = useState("vitals")
  const [sosActive, setSosActive] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.3 }}
      className="space-y-4"
    >
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-xl font-semibold text-red-700">PulseChain</h2>
          <p className="text-gray-600">Emergency assistance</p>
        </div>
        <SosButton active={sosActive} setActive={setSosActive} />
      </div>

      <AnimatePresence>
        {sosActive && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="bg-red-50 border-red-200">
              <CardContent className="p-4">
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-75"></div>
                    <div className="relative rounded-full bg-red-600 h-10 w-10 flex items-center justify-center text-white">
                      SOS
                    </div>
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-red-700">Emergency Mode Active</h3>
                    <p className="text-sm text-red-600">
                      Your location is being shared with emergency contacts. Stay calm and wait for assistance.
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    className="border-red-300 text-red-700 hover:bg-red-100"
                    onClick={() => setSosActive(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      <Tabs defaultValue="vitals" className="w-full" onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-3 mb-4">
          <TabsTrigger value="vitals">Vitals</TabsTrigger>
          <TabsTrigger value="contacts">Contacts</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="vitals" className="space-y-4">
          <VitalsMonitor />
        </TabsContent>

        <TabsContent value="contacts" className="space-y-4">
          <EmergencyContacts />
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <AlertsFeed />
        </TabsContent>
      </Tabs>
    </motion.div>
  )
}
