"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { SymptomChecker } from "./symptom-checker"
import { MedicineReminder } from "./medicine-reminder"
import { AppointmentBooking } from "./appointment-booking"
import { motion } from "framer-motion"

export default function MediMate() {
  const [activeTab, setActiveTab] = useState("symptoms")

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.3 }}
      className="space-y-4"
    >
      <h2 className="text-xl font-semibold text-teal-700">MediMate</h2>
      <p className="text-gray-600">Your personal medical assistant</p>

      <Tabs defaultValue="symptoms" className="w-full" onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-3 mb-4">
          <TabsTrigger value="symptoms">Symptoms</TabsTrigger>
          <TabsTrigger value="reminders">Reminders</TabsTrigger>
          <TabsTrigger value="appointments">Appointments</TabsTrigger>
        </TabsList>

        <TabsContent value="symptoms" className="space-y-4">
          <SymptomChecker />
        </TabsContent>

        <TabsContent value="reminders" className="space-y-4">
          <MedicineReminder />
        </TabsContent>

        <TabsContent value="appointments" className="space-y-4">
          <AppointmentBooking />
        </TabsContent>
      </Tabs>
    </motion.div>
  )
}
