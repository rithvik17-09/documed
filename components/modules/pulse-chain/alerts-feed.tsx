"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Loader2, AlertTriangle, Bell, Info } from "lucide-react"
import { motion } from "framer-motion"

type Alert = {
  id: string
  type: "emergency" | "warning" | "info"
  title: string
  message: string
  time: string
  read: boolean
}

export function AlertsFeed() {
  const [isLoading, setIsLoading] = useState(true)
  const [alerts, setAlerts] = useState<Alert[]>([])

  useEffect(() => {
    // Simulate API call to Gemini for alerts
    const fetchAlerts = async () => {
      setIsLoading(true)

      // Simulate delay
      await new Promise((resolve) => setTimeout(resolve, 1500))

      // Mock alerts
      const mockAlerts: Alert[] = [
        {
          id: "1",
          type: "emergency",
          title: "Severe Weather Alert",
          message: "Flash flood warning in your area. Seek higher ground if needed.",
          time: "10 minutes ago",
          read: false,
        },
        {
          id: "2",
          type: "warning",
          title: "Air Quality Warning",
          message: "Poor air quality reported in your area. Limit outdoor activities if you have respiratory issues.",
          time: "1 hour ago",
          read: false,
        },
        {
          id: "3",
          type: "info",
          title: "COVID-19 Update",
          message: "New vaccination center opened near your location. Check local health department for details.",
          time: "3 hours ago",
          read: true,
        },
      ]

      setAlerts(mockAlerts)
      setIsLoading(false)
    }

    fetchAlerts()
  }, [])

  const markAsRead = (id: string) => {
    setAlerts((prev) => prev.map((alert) => (alert.id === id ? { ...alert, read: true } : alert)))
  }

  const markAllAsRead = () => {
    setAlerts((prev) => prev.map((alert) => ({ ...alert, read: true })))
  }

  const getAlertIcon = (type: Alert["type"]) => {
    switch (type) {
      case "emergency":
        return <AlertTriangle className="h-5 w-5 text-red-500" />
      case "warning":
        return <Bell className="h-5 w-5 text-amber-500" />
      case "info":
        return <Info className="h-5 w-5 text-blue-500" />
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Health & Safety Alerts</h3>
        {alerts.some((alert) => !alert.read) && (
          <Button variant="outline" size="sm" onClick={markAllAsRead}>
            Mark all as read
          </Button>
        )}
      </div>

      {isLoading ? (
        <Card className="bg-gray-50">
          <CardContent className="p-8 flex justify-center items-center">
            <Loader2 className="h-8 w-8 animate-spin text-red-500" />
          </CardContent>
        </Card>
      ) : alerts.length > 0 ? (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Card
                className={`
                border-l-4 
                ${
                  alert.type === "emergency"
                    ? "border-l-red-500 bg-red-50"
                    : alert.type === "warning"
                      ? "border-l-amber-500 bg-amber-50"
                      : "border-l-blue-500 bg-blue-50"
                }
                ${alert.read ? "opacity-70" : ""}
              `}
              >
                <CardContent className="p-4">
                  <div className="flex gap-3">
                    <div className="flex-shrink-0 mt-1">{getAlertIcon(alert.type)}</div>
                    <div className="flex-1">
                      <div className="flex justify-between items-start">
                        <h4 className="font-medium">{alert.title}</h4>
                        <span className="text-xs text-gray-500">{alert.time}</span>
                      </div>
                      <p className="text-sm text-gray-700 mt-1">{alert.message}</p>
                      {!alert.read && (
                        <Button variant="link" className="p-0 h-auto text-sm mt-2" onClick={() => markAsRead(alert.id)}>
                          Mark as read
                        </Button>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      ) : (
        <Card className="bg-gray-50">
          <CardContent className="p-8 text-center text-gray-500">
            No alerts at this time. You'll be notified when new alerts are available.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
