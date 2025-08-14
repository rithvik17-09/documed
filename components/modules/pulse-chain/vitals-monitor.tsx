"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { RefreshCw, Heart, Droplet, Thermometer } from "lucide-react"
import { motion } from "framer-motion"

export function VitalsMonitor() {
  const [vitals, setVitals] = useState({
    heartRate: 72,
    oxygen: 98,
    temperature: 98.6,
  })
  const [isLoading, setIsLoading] = useState(false)
  const [pulseAnimation, setPulseAnimation] = useState(false)

  // Generate random vitals
  const generateVitals = () => {
    setIsLoading(true)
    setPulseAnimation(true)

    setTimeout(() => {
      setVitals({
        heartRate: Math.floor(Math.random() * (100 - 60) + 60),
        oxygen: Math.floor(Math.random() * (100 - 94) + 94),
        temperature: Number.parseFloat((Math.random() * (99.5 - 97.5) + 97.5).toFixed(1)),
      })
      setIsLoading(false)

      // Stop pulse animation after 2 seconds
      setTimeout(() => {
        setPulseAnimation(false)
      }, 2000)
    }, 1000)
  }

  // Initial vitals generation
  useEffect(() => {
    generateVitals()
  }, [])

  const getHeartRateStatus = (rate: number) => {
    if (rate < 60) return "low"
    if (rate > 100) return "high"
    return "normal"
  }

  const getOxygenStatus = (level: number) => {
    if (level < 95) return "low"
    return "normal"
  }

  const getTemperatureStatus = (temp: number) => {
    if (temp < 97.8) return "low"
    if (temp > 99.1) return "high"
    return "normal"
  }

  const heartRateStatus = getHeartRateStatus(vitals.heartRate)
  const oxygenStatus = getOxygenStatus(vitals.oxygen)
  const temperatureStatus = getTemperatureStatus(vitals.temperature)

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Vitals Monitor</h3>
        <Button variant="outline" size="sm" onClick={generateVitals} disabled={isLoading}>
          {isLoading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
        </Button>
      </div>

      <div className="grid gap-4">
        <Card
          className={`
          ${
            heartRateStatus === "normal"
              ? "bg-green-50 border-green-100"
              : heartRateStatus === "high"
                ? "bg-red-50 border-red-100"
                : "bg-blue-50 border-blue-100"
          }
        `}
        >
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <motion.div
                  animate={pulseAnimation ? { scale: [1, 1.2, 1] } : {}}
                  transition={{ repeat: 5, duration: 0.5 }}
                >
                  <Heart
                    className={`
                    h-6 w-6 
                    ${
                      heartRateStatus === "normal"
                        ? "text-green-500"
                        : heartRateStatus === "high"
                          ? "text-red-500"
                          : "text-blue-500"
                    }
                  `}
                  />
                </motion.div>
                <div>
                  <h4 className="font-medium">Heart Rate</h4>
                  <p className="text-sm text-gray-500">Beats per minute</p>
                </div>
              </div>
              <div className="text-right">
                <span className="text-2xl font-bold">{vitals.heartRate}</span>
                <span className="text-sm ml-1">BPM</span>
              </div>
            </div>
            <Progress
              value={(vitals.heartRate / 200) * 100}
              className={`h-2 mt-3 ${
                heartRateStatus === "normal"
                  ? "bg-green-100"
                  : heartRateStatus === "high"
                    ? "bg-red-100"
                    : "bg-blue-100"
              }`}
            />
            <p
              className={`text-sm mt-2 ${
                heartRateStatus === "normal"
                  ? "text-green-600"
                  : heartRateStatus === "high"
                    ? "text-red-600"
                    : "text-blue-600"
              }`}
            >
              {heartRateStatus === "normal"
                ? "Normal heart rate"
                : heartRateStatus === "high"
                  ? "Elevated heart rate"
                  : "Low heart rate"}
            </p>
          </CardContent>
        </Card>

        <Card
          className={`
          ${oxygenStatus === "normal" ? "bg-green-50 border-green-100" : "bg-red-50 border-red-100"}
        `}
        >
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Droplet
                  className={`
                  h-6 w-6 
                  ${oxygenStatus === "normal" ? "text-green-500" : "text-red-500"}
                `}
                />
                <div>
                  <h4 className="font-medium">Blood Oxygen</h4>
                  <p className="text-sm text-gray-500">SpO2 percentage</p>
                </div>
              </div>
              <div className="text-right">
                <span className="text-2xl font-bold">{vitals.oxygen}</span>
                <span className="text-sm ml-1">%</span>
              </div>
            </div>
            <Progress
              value={vitals.oxygen}
              className={`h-2 mt-3 ${oxygenStatus === "normal" ? "bg-green-100" : "bg-red-100"}`}
            />
            <p className={`text-sm mt-2 ${oxygenStatus === "normal" ? "text-green-600" : "text-red-600"}`}>
              {oxygenStatus === "normal" ? "Normal oxygen level" : "Low oxygen level"}
            </p>
          </CardContent>
        </Card>

        <Card
          className={`
          ${
            temperatureStatus === "normal"
              ? "bg-green-50 border-green-100"
              : temperatureStatus === "high"
                ? "bg-red-50 border-red-100"
                : "bg-blue-50 border-blue-100"
          }
        `}
        >
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Thermometer
                  className={`
                  h-6 w-6 
                  ${
                    temperatureStatus === "normal"
                      ? "text-green-500"
                      : temperatureStatus === "high"
                        ? "text-red-500"
                        : "text-blue-500"
                  }
                `}
                />
                <div>
                  <h4 className="font-medium">Temperature</h4>
                  <p className="text-sm text-gray-500">Degrees Fahrenheit</p>
                </div>
              </div>
              <div className="text-right">
                <span className="text-2xl font-bold">{vitals.temperature}</span>
                <span className="text-sm ml-1">°F</span>
              </div>
            </div>
            <Progress
              value={((vitals.temperature - 95) / 10) * 100}
              className={`h-2 mt-3 ${
                temperatureStatus === "normal"
                  ? "bg-green-100"
                  : temperatureStatus === "high"
                    ? "bg-red-100"
                    : "bg-blue-100"
              }`}
            />
            <p
              className={`text-sm mt-2 ${
                temperatureStatus === "normal"
                  ? "text-green-600"
                  : temperatureStatus === "high"
                    ? "text-red-600"
                    : "text-blue-600"
              }`}
            >
              {temperatureStatus === "normal"
                ? "Normal temperature"
                : temperatureStatus === "high"
                  ? "Elevated temperature"
                  : "Low temperature"}
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
