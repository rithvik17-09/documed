"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { CalendarIcon, Clock, MapPin } from "lucide-react"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { format } from "date-fns"
import { cn } from "@/lib/utils"
import { motion } from "framer-motion"

type Appointment = {
  id: string
  doctorType: string
  date: Date
  time: string
  location: string
  notes: string
}

export function AppointmentBooking() {
  const [date, setDate] = useState<Date>()
  const [appointments, setAppointments] = useState<Appointment[]>([
    {
      id: "1",
      doctorType: "General Practitioner",
      date: new Date(2025, 4, 15),
      time: "10:00",
      location: "City Health Clinic",
      notes: "Annual checkup",
    },
  ])

  const [formData, setFormData] = useState({
    doctorType: "",
    time: "",
    location: "",
    notes: "",
  })

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!date || !formData.doctorType || !formData.time) return

    setIsSubmitting(true)

    // Simulate API call
    setTimeout(() => {
      setAppointments((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          date: date,
          ...formData,
        },
      ])

      // Reset form
      setDate(undefined)
      setFormData({
        doctorType: "",
        time: "",
        location: "",
        notes: "",
      })

      setIsSubmitting(false)
      setIsSuccess(true)

      // Reset success message after 3 seconds
      setTimeout(() => setIsSuccess(false), 3000)
    }, 1000)
  }

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Upcoming Appointments</h3>

        {appointments.map((appointment) => (
          <motion.div
            key={appointment.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="bg-gradient-to-r from-blue-50 to-teal-50 border-blue-100">
              <CardContent className="p-4">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-semibold text-blue-700">{appointment.doctorType}</h4>
                    <div className="flex items-center text-sm text-gray-600 mt-1">
                      <CalendarIcon className="h-3 w-3 mr-1" />
                      {format(appointment.date, "PPP")}
                      <Clock className="h-3 w-3 ml-2 mr-1" />
                      {appointment.time}
                    </div>
                    <div className="flex items-center text-sm text-gray-600 mt-1">
                      <MapPin className="h-3 w-3 mr-1" />
                      {appointment.location}
                    </div>
                    {appointment.notes && <p className="text-sm text-gray-600 mt-2 italic">"{appointment.notes}"</p>}
                  </div>
                  <Button variant="outline" size="sm" className="bg-white">
                    Reschedule
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}

        {appointments.length === 0 && (
          <Card className="bg-gray-50">
            <CardContent className="p-8 text-center text-gray-500">
              No upcoming appointments. Book your first appointment below.
            </CardContent>
          </Card>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Book New Appointment</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid gap-2">
              <Label htmlFor="doctorType">Doctor Type</Label>
              <Select
                value={formData.doctorType}
                onValueChange={(value) => setFormData((prev) => ({ ...prev, doctorType: value }))}
                required
              >
                <SelectTrigger id="doctorType">
                  <SelectValue placeholder="Select specialist" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="General Practitioner">General Practitioner</SelectItem>
                  <SelectItem value="Cardiologist">Cardiologist</SelectItem>
                  <SelectItem value="Dermatologist">Dermatologist</SelectItem>
                  <SelectItem value="Neurologist">Neurologist</SelectItem>
                  <SelectItem value="Pediatrician">Pediatrician</SelectItem>
                  <SelectItem value="Psychiatrist">Psychiatrist</SelectItem>
                  <SelectItem value="Orthopedist">Orthopedist</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-2">
              <Label>Date</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn("w-full justify-start text-left font-normal", !date && "text-muted-foreground")}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {date ? format(date, "PPP") : <span>Pick a date</span>}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    mode="single"
                    selected={date}
                    onSelect={setDate}
                    initialFocus
                    disabled={(date) => date < new Date()}
                  />
                </PopoverContent>
              </Popover>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="time">Time</Label>
              <Select
                value={formData.time}
                onValueChange={(value) => setFormData((prev) => ({ ...prev, time: value }))}
                required
              >
                <SelectTrigger id="time">
                  <SelectValue placeholder="Select time" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="09:00">9:00 AM</SelectItem>
                  <SelectItem value="10:00">10:00 AM</SelectItem>
                  <SelectItem value="11:00">11:00 AM</SelectItem>
                  <SelectItem value="13:00">1:00 PM</SelectItem>
                  <SelectItem value="14:00">2:00 PM</SelectItem>
                  <SelectItem value="15:00">3:00 PM</SelectItem>
                  <SelectItem value="16:00">4:00 PM</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="location">Location</Label>
              <Input
                id="location"
                value={formData.location}
                onChange={(e) => setFormData((prev) => ({ ...prev, location: e.target.value }))}
                placeholder="Clinic or hospital name"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="notes">Notes (Optional)</Label>
              <Textarea
                id="notes"
                value={formData.notes}
                onChange={(e) => setFormData((prev) => ({ ...prev, notes: e.target.value }))}
                placeholder="Any additional information for the doctor"
              />
            </div>

            <Button
              type="submit"
              className="w-full bg-gradient-to-r from-teal-500 to-blue-500 hover:from-teal-600 hover:to-blue-600"
              disabled={isSubmitting || !date || !formData.doctorType || !formData.time}
            >
              {isSubmitting ? "Booking..." : "Book Appointment"}
            </Button>

            {isSuccess && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3 bg-green-100 text-green-700 rounded-md text-center"
              >
                Appointment booked successfully!
              </motion.div>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
