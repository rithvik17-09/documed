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
      <h2 className="text-xl font-semibold text-teal-700 dark:text-teal-400 transition-colors">MediMate</h2>
      <p className="text-gray-600 dark:text-gray-300 transition-colors">Your personal medical assistant</p>

      <Tabs defaultValue="symptoms" className="w-full" onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-3 mb-4 bg-gray-100 dark:bg-gray-800 transition-colors">
          <TabsTrigger value="symptoms" className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 transition-colors">Symptoms</TabsTrigger>
          <TabsTrigger value="reminders" className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 transition-colors">Reminders</TabsTrigger>
          <TabsTrigger value="appointments" className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 transition-colors">Appointments</TabsTrigger>
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
