"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Clock, Plus, Trash2, Bell, Check } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

type Reminder = {
  id: string
  medicine: string
  time: string
  dosage: string
  frequency: string
  completed: boolean
}

export function MedicineReminder() {
  const [reminders, setReminders] = useState<Reminder[]>([
    {
      id: "1",
      medicine: "Ibuprofen",
      time: "08:00",
      dosage: "400mg",
      frequency: "daily",
      completed: false,
    },
    {
      id: "2",
      medicine: "Vitamin D",
      time: "09:00",
      dosage: "1000 IU",
      frequency: "daily",
      completed: true,
    },
  ])

  const [newReminder, setNewReminder] = useState<Omit<Reminder, "id" | "completed">>({
    medicine: "",
    time: "",
    dosage: "",
    frequency: "daily",
  })

  const addReminder = () => {
    if (!newReminder.medicine || !newReminder.time) return

    setReminders((prev) => [
      ...prev,
      {
        ...newReminder,
        id: Date.now().toString(),
        completed: false,
      },
    ])

    setNewReminder({
      medicine: "",
      time: "",
      dosage: "",
      frequency: "daily",
    })
  }

  const toggleComplete = (id: string) => {
    setReminders((prev) =>
      prev.map((reminder) => (reminder.id === id ? { ...reminder, completed: !reminder.completed } : reminder)),
    )
  }

  const deleteReminder = (id: string) => {
    setReminders((prev) => prev.filter((reminder) => reminder.id !== id))
  }

  const getTimeStatus = (time: string) => {
    const [hours, minutes] = time.split(":").map(Number)
    const reminderTime = new Date()
    reminderTime.setHours(hours, minutes, 0)

    const now = new Date()
    const diffMs = reminderTime.getTime() - now.getTime()

    if (diffMs < 0) {
      return "past"
    } else if (diffMs < 3600000) {
      // Less than 1 hour
      return "soon"
    } else {
      return "upcoming"
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Medicine Reminders</h3>
        <Dialog>
          <DialogTrigger asChild>
            <Button
              size="sm"
              className="bg-gradient-to-r from-teal-500 to-blue-500 hover:from-teal-600 hover:to-blue-600"
            >
              <Plus className="h-4 w-4 mr-1" /> Add Reminder
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add Medicine Reminder</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="medicine">Medicine Name</Label>
                <Input
                  id="medicine"
                  value={newReminder.medicine}
                  onChange={(e) => setNewReminder((prev) => ({ ...prev, medicine: e.target.value }))}
                  placeholder="e.g., Ibuprofen"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="time">Time</Label>
                <Input
                  id="time"
                  type="time"
                  value={newReminder.time}
                  onChange={(e) => setNewReminder((prev) => ({ ...prev, time: e.target.value }))}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="dosage">Dosage</Label>
                <Input
                  id="dosage"
                  value={newReminder.dosage}
                  onChange={(e) => setNewReminder((prev) => ({ ...prev, dosage: e.target.value }))}
                  placeholder="e.g., 400mg"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="frequency">Frequency</Label>
                <Select
                  value={newReminder.frequency}
                  onValueChange={(value) => setNewReminder((prev) => ({ ...prev, frequency: value }))}
                >
                  <SelectTrigger id="frequency">
                    <SelectValue placeholder="Select frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="daily">Daily</SelectItem>
                    <SelectItem value="twice-daily">Twice Daily</SelectItem>
                    <SelectItem value="weekly">Weekly</SelectItem>
                    <SelectItem value="as-needed">As Needed</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline">Cancel</Button>
              </DialogClose>
              <DialogClose asChild>
                <Button onClick={addReminder}>Save Reminder</Button>
              </DialogClose>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="space-y-3">
        <AnimatePresence>
          {reminders.map((reminder) => {
            const timeStatus = getTimeStatus(reminder.time)

            return (
              <motion.div
                key={reminder.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -100 }}
                transition={{ duration: 0.3 }}
              >
                <Card
                  className={`
                  border-l-4 
                  ${
                    reminder.completed
                      ? "border-l-green-500 bg-green-50"
                      : timeStatus === "past"
                        ? "border-l-red-500 bg-red-50"
                        : timeStatus === "soon"
                          ? "border-l-yellow-500 bg-yellow-50"
                          : "border-l-blue-500 bg-blue-50"
                  }
                `}
                >
                  <CardContent className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Button
                        variant="outline"
                        size="icon"
                        className={`rounded-full h-8 w-8 ${
                          reminder.completed ? "bg-green-100 text-green-600 border-green-200" : "bg-white"
                        }`}
                        onClick={() => toggleComplete(reminder.id)}
                      >
                        {reminder.completed ? <Check className="h-4 w-4" /> : <Bell className="h-4 w-4" />}
                      </Button>
                      <div>
                        <h4 className="font-medium">{reminder.medicine}</h4>
                        <div className="flex items-center text-sm text-gray-500">
                          <Clock className="h-3 w-3 mr-1" />
                          {reminder.time} • {reminder.dosage} • {reminder.frequency.replace("-", " ")}
                        </div>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-gray-400 hover:text-red-500"
                      onClick={() => deleteReminder(reminder.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </AnimatePresence>

        {reminders.length === 0 && (
          <Card className="bg-gray-50">
            <CardContent className="p-8 text-center text-gray-500">
              No medicine reminders yet. Add your first reminder to get started.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
