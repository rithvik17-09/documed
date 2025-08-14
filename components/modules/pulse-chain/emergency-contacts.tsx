"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
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
import { Phone, MessageSquare, User, Plus, Trash2 } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

type Contact = {
  id: string
  name: string
  phone: string
  relationship: string
}

export function EmergencyContacts() {
  const [contacts, setContacts] = useState<Contact[]>([
    {
      id: "1",
      name: "John Doe",
      phone: "(555) 123-4567",
      relationship: "Family",
    },
    {
      id: "2",
      name: "Jane Smith",
      phone: "(555) 987-6543",
      relationship: "Friend",
    },
  ])

  const [newContact, setNewContact] = useState({
    name: "",
    phone: "",
    relationship: "",
  })

  const addContact = () => {
    if (!newContact.name || !newContact.phone) return

    setContacts((prev) => [
      ...prev,
      {
        ...newContact,
        id: Date.now().toString(),
      },
    ])

    setNewContact({
      name: "",
      phone: "",
      relationship: "",
    })
  }

  const deleteContact = (id: string) => {
    setContacts((prev) => prev.filter((contact) => contact.id !== id))
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Emergency Contacts</h3>
        <Dialog>
          <DialogTrigger asChild>
            <Button
              size="sm"
              className="bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600"
            >
              <Plus className="h-4 w-4 mr-1" /> Add Contact
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add Emergency Contact</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  value={newContact.name}
                  onChange={(e) => setNewContact((prev) => ({ ...prev, name: e.target.value }))}
                  placeholder="Contact name"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="phone">Phone Number</Label>
                <Input
                  id="phone"
                  value={newContact.phone}
                  onChange={(e) => setNewContact((prev) => ({ ...prev, phone: e.target.value }))}
                  placeholder="(555) 123-4567"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="relationship">Relationship</Label>
                <Input
                  id="relationship"
                  value={newContact.relationship}
                  onChange={(e) => setNewContact((prev) => ({ ...prev, relationship: e.target.value }))}
                  placeholder="e.g., Family, Friend, Doctor"
                />
              </div>
            </div>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline">Cancel</Button>
              </DialogClose>
              <DialogClose asChild>
                <Button onClick={addContact}>Save Contact</Button>
              </DialogClose>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <AnimatePresence>
        {contacts.map((contact) => (
          <motion.div
            key={contact.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="bg-white">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center rounded-full bg-red-100 w-10 h-10 text-red-600 flex-shrink-0">
                    <User className="h-5 w-5" />
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="font-medium">{contact.name}</h4>
                        <div className="text-sm text-gray-500">
                          {contact.phone} • {contact.relationship}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="icon"
                          className="h-8 w-8 rounded-full bg-green-50 text-green-600 border-green-200 hover:bg-green-100"
                        >
                          <Phone className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="icon"
                          className="h-8 w-8 rounded-full bg-blue-50 text-blue-600 border-blue-200 hover:bg-blue-100"
                        >
                          <MessageSquare className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="icon"
                          className="h-8 w-8 rounded-full bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100 hover:text-red-600"
                          onClick={() => deleteContact(contact.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </AnimatePresence>

      {contacts.length === 0 && (
        <Card className="bg-gray-50">
          <CardContent className="p-8 text-center text-gray-500">
            No emergency contacts added yet. Add your first contact to get started.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
