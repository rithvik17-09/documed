"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { motion } from "framer-motion"

interface SosButtonProps {
  active: boolean
  setActive: (active: boolean) => void
}

export function SosButton({ active, setActive }: SosButtonProps) {
  const [isConfirming, setIsConfirming] = useState(false)

  const activateSos = () => {
    setActive(true)
    setIsConfirming(false)
  }

  return (
    <AlertDialog open={isConfirming} onOpenChange={setIsConfirming}>
      <AlertDialogTrigger asChild>
        <Button
          variant={active ? "outline" : "default"}
          size="lg"
          className={`
            rounded-full font-bold
            ${
              active
                ? "border-red-500 text-red-500 hover:bg-red-50"
                : "bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600"
            }
          `}
          onClick={() => {
            if (active) {
              setActive(false)
            }
          }}
        >
          <motion.span
            animate={active ? { scale: [1, 1.1, 1] } : {}}
            transition={{ repeat: Number.POSITIVE_INFINITY, duration: 2 }}
          >
            {active ? "SOS ACTIVE" : "SOS"}
          </motion.span>
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Activate Emergency SOS?</AlertDialogTitle>
          <AlertDialogDescription>
            This will alert your emergency contacts and share your current location with them. Only use this in case of
            a genuine emergency.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={activateSos} className="bg-red-500 hover:bg-red-600">
            Activate SOS
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
