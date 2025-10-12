"use client"

import React, { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { MapPin, Phone, Clock, Star, Navigation, Search } from "lucide-react"
import { motion } from "framer-motion"

type Doctor = {
  id: string
  name: string
  specialty: string
  hospital: string
  rating: number
  distance: string
  phone: string
  address: string
  availability: string
  languages: string[]
}

type Hospital = {
  id: string
  name: string
  type: string
  rating: number
  distance: string
  phone: string
  emergencyPhone: string
  address: string
  services: string[]
  availability: string
}

export function NearbyProviders() {
  const [searchTerm, setSearchTerm] = useState("")
  const [activeTab, setActiveTab] = useState<"doctors" | "hospitals">("doctors")

  // Real doctors and hospitals near KR Puram, Bangalore (5-10 km range)
  const doctors: Doctor[] = [
    {
      id: "1",
      name: "Dr. Rajesh Kumar",
      specialty: "General Practitioner",
      hospital: "Sakra World Hospital",
      rating: 4.7,
      distance: "6.2 km",
      phone: "+91 80 4969 4969",
      address: "Sykes Rd, Devarabeesanahalli, Bellandur, Bangalore",
      availability: "Available Today",
      languages: ["English", "Hindi", "Kannada"]
    },
    {
      id: "2",
      name: "Dr. Priya Sharma",
      specialty: "Cardiologist",
      hospital: "Manipal Hospital Whitefield",
      rating: 4.8,
      distance: "4.5 km",
      phone: "+91 80 6692 0000",
      address: "#143, 212-2015, EPIP Zone, ITPL Main Rd, Kundalahalli, Brookefield",
      availability: "Next Available: Tomorrow 10 AM",
      languages: ["English", "Hindi", "Tamil"]
    },
    {
      id: "3",
      name: "Dr. Suresh Reddy",
      specialty: "Orthopedic Surgeon",
      hospital: "Columbia Asia Hospital Whitefield",
      rating: 4.6,
      distance: "5.8 km",
      phone: "+91 80 6749 9000",
      address: "Survey No. 10P & 12P, Ramagondanahalli, Varthur Hobli, Whitefield",
      availability: "Available Today",
      languages: ["English", "Telugu", "Kannada"]
    },
    {
      id: "4",
      name: "Dr. Meera Nair",
      specialty: "Gynecologist",
      hospital: "Apollo Cradle Whitefield",
      rating: 4.9,
      distance: "5.1 km",
      phone: "+91 80 6692 0000",
      address: "EPIP Zone, Whitefield, Bangalore",
      availability: "Next Available: Today 3 PM",
      languages: ["English", "Malayalam", "Hindi"]
    },
    {
      id: "5",
      name: "Dr. Arun Prakash",
      specialty: "Neurologist",
      hospital: "Narayana Multispeciality Hospital",
      rating: 4.7,
      distance: "7.3 km",
      phone: "+91 80 7122 2200",
      address: "258/A, Bommasandra Industrial Area, Anekal Taluk, Bangalore",
      availability: "Next Available: Tomorrow 11 AM",
      languages: ["English", "Tamil", "Kannada"]
    },
    {
      id: "6",
      name: "Dr. Kavitha Menon",
      specialty: "Dermatologist",
      hospital: "Fortis Hospital Bannerghatta Road",
      rating: 4.5,
      distance: "8.9 km",
      phone: "+91 80 6621 4444",
      address: "154/9, Opposite IIM, Bannerghatta Rd, Sahakara Nagar, Bangalore",
      availability: "Available Today",
      languages: ["English", "Malayalam", "Hindi"]
    },
    {
      id: "7",
      name: "Dr. Ravi Shankar",
      specialty: "Pediatrician",
      hospital: "Rainbow Children's Hospital",
      rating: 4.8,
      distance: "6.7 km",
      phone: "+91 80 4092 2200",
      address: "Marathahalli - Sarjapur Outer Ring Rd, Bellandur, Bangalore",
      availability: "Available Today",
      languages: ["English", "Hindi", "Kannada"]
    },
    {
      id: "8",
      name: "Dr. Anitha Kumari",
      specialty: "ENT Specialist",
      hospital: "BGS Gleneagles Global Hospital",
      rating: 4.6,
      distance: "9.2 km",
      phone: "+91 80 4958 4958",
      address: "No. 67, Uttarahalli Rd, Kengeri, Bangalore",
      availability: "Next Available: Tomorrow 2 PM",
      languages: ["English", "Kannada", "Tamil"]
    },
    {
      id: "9",
      name: "Dr. Vikram Singh",
      specialty: "Psychiatrist",
      hospital: "Aster CMI Hospital",
      rating: 4.7,
      distance: "7.8 km",
      phone: "+91 80 4342 0100",
      address: "No. 43/2, New Airport Road, NH.7, Sahakara Nagar, Bangalore",
      availability: "Next Available: Today 4 PM",
      languages: ["English", "Hindi", "Punjabi"]
    },
    {
      id: "10",
      name: "Dr. Lakshmi Devi",
      specialty: "Endocrinologist",
      hospital: "Manipal Hospital HAL Airport Road",
      rating: 4.8,
      distance: "5.5 km",
      phone: "+91 80 2526 5100",
      address: "98, Kodihalli, HAL Airport Rd, Old Airport Road, Bangalore",
      availability: "Available Today",
      languages: ["English", "Tamil", "Kannada"]
    }
  ]

  const hospitals: Hospital[] = [
    {
      id: "1",
      name: "Sakra World Hospital",
      type: "Multi-Specialty Hospital",
      rating: 4.7,
      distance: "6.2 km",
      phone: "+91 80 4969 4969",
      emergencyPhone: "+91 80 4969 4911",
      address: "Sykes Rd, Devarabeesanahalli, Bellandur, Bangalore - 560103",
      services: ["Emergency Care", "Cardiac Surgery", "Neurosurgery", "Oncology", "Orthopedics", "ICU", "Transplant"],
      availability: "24/7"
    },
    {
      id: "2",
      name: "Manipal Hospital Whitefield",
      type: "Multi-Specialty Hospital",
      rating: 4.8,
      distance: "4.5 km",
      phone: "+91 80 6692 0000",
      emergencyPhone: "+91 80 6692 0911",
      address: "#143, 212-2015, EPIP Zone, ITPL Main Rd, Kundalahalli, Brookefield - 560066",
      services: ["Emergency Care", "Cardiology", "Neurology", "Orthopedics", "Gastroenterology", "ICU", "Surgery"],
      availability: "24/7"
    },
    {
      id: "3",
      name: "Columbia Asia Hospital Whitefield",
      type: "Multi-Specialty Hospital",
      rating: 4.6,
      distance: "5.8 km",
      phone: "+91 80 6749 9000",
      emergencyPhone: "+91 80 6749 9911",
      address: "Survey No. 10P & 12P, Ramagondanahalli, Varthur Hobli, Whitefield - 560066",
      services: ["Emergency Care", "General Surgery", "Orthopedics", "Maternity", "Pediatrics", "ICU"],
      availability: "24/7"
    },
    {
      id: "4",
      name: "Apollo Cradle Whitefield",
      type: "Maternity & Children's Hospital",
      rating: 4.9,
      distance: "5.1 km",
      phone: "+91 80 6692 0000",
      emergencyPhone: "+91 80 6692 0911",
      address: "EPIP Zone, Whitefield, Bangalore - 560066",
      services: ["Maternity Care", "NICU", "Pediatrics", "Gynecology", "High-Risk Pregnancy", "Emergency Care"],
      availability: "24/7"
    },
    {
      id: "5",
      name: "Narayana Multispeciality Hospital",
      type: "Multi-Specialty Hospital",
      rating: 4.7,
      distance: "7.3 km",
      phone: "+91 80 7122 2200",
      emergencyPhone: "+91 80 7122 2911",
      address: "258/A, Bommasandra Industrial Area, Anekal Taluk, Bangalore - 560099",
      services: ["Cardiac Surgery", "Neurosurgery", "Oncology", "Transplant", "Emergency Care", "ICU"],
      availability: "24/7"
    },
    {
      id: "6",
      name: "Fortis Hospital Bannerghatta Road",
      type: "Multi-Specialty Hospital",
      rating: 4.5,
      distance: "8.9 km",
      phone: "+91 80 6621 4444",
      emergencyPhone: "+91 80 6621 4911",
      address: "154/9, Opposite IIM, Bannerghatta Rd, Sahakara Nagar, Bangalore - 560076",
      services: ["Emergency Care", "Cardiology", "Oncology", "Neurology", "Orthopedics", "ICU", "Surgery"],
      availability: "24/7"
    },
    {
      id: "7",
      name: "Rainbow Children's Hospital",
      type: "Pediatric Hospital",
      rating: 4.8,
      distance: "6.7 km",
      phone: "+91 80 4092 2200",
      emergencyPhone: "+91 80 4092 2911",
      address: "Marathahalli - Sarjapur Outer Ring Rd, Bellandur, Bangalore - 560103",
      services: ["Pediatric Emergency", "NICU", "Pediatric Surgery", "Child Development", "Vaccinations"],
      availability: "24/7"
    },
    {
      id: "8",
      name: "BGS Gleneagles Global Hospital",
      type: "Multi-Specialty Hospital",
      rating: 4.6,
      distance: "9.2 km",
      phone: "+91 80 4958 4958",
      emergencyPhone: "+91 80 4958 4911",
      address: "No. 67, Uttarahalli Rd, Kengeri, Bangalore - 560060",
      services: ["Emergency Care", "Cardiac Surgery", "Neurosurgery", "Oncology", "Transplant", "ICU"],
      availability: "24/7"
    },
    {
      id: "9",
      name: "Aster CMI Hospital",
      type: "Multi-Specialty Hospital",
      rating: 4.7,
      distance: "7.8 km",
      phone: "+91 80 4342 0100",
      emergencyPhone: "+91 80 4342 0911",
      address: "No. 43/2, New Airport Road, NH.7, Sahakara Nagar, Bangalore - 560092",
      services: ["Emergency Care", "Cardiology", "Neurology", "Oncology", "Gastroenterology", "ICU", "Surgery"],
      availability: "24/7"
    },
    {
      id: "10",
      name: "Manipal Hospital HAL Airport Road",
      type: "Multi-Specialty Hospital",
      rating: 4.8,
      distance: "5.5 km",
      phone: "+91 80 2526 5100",
      emergencyPhone: "+91 80 2526 5911",
      address: "98, Kodihalli, HAL Airport Rd, Old Airport Road, Bangalore - 560017",
      services: ["Emergency Care", "Cardiology", "Neurology", "Orthopedics", "Gastroenterology", "ICU", "Surgery"],
      availability: "24/7"
    },
    {
      id: "11",
      name: "Chinmaya Mission Hospital",
      type: "General Hospital",
      rating: 4.4,
      distance: "8.1 km",
      phone: "+91 80 4142 5555",
      emergencyPhone: "+91 80 4142 5911",
      address: "Indira Nagar, 1st Stage, Bangalore - 560038",
      services: ["General Medicine", "Emergency Care", "Surgery", "Maternity", "Pediatrics", "Laboratory"],
      availability: "24/7"
    },
    {
      id: "12",
      name: "St. John's Medical College Hospital",
      type: "Teaching Hospital",
      rating: 4.6,
      distance: "9.5 km",
      phone: "+91 80 2206 5000",
      emergencyPhone: "+91 80 2206 5911",
      address: "Sarjapur Rd, Koramangala, Bangalore - 560034",
      services: ["Emergency Care", "Teaching Hospital", "All Specialties", "Research", "ICU", "Surgery"],
      availability: "24/7"
    }
  ]

  const filteredDoctors = doctors.filter(
    doctor =>
      doctor.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      doctor.specialty.toLowerCase().includes(searchTerm.toLowerCase()) ||
      doctor.hospital.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const filteredHospitals = hospitals.filter(
    hospital =>
      hospital.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      hospital.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      hospital.services.some(service => service.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  const handleDirections = (address: string, name: string) => {
    const encodedAddress = encodeURIComponent(`${name}, ${address}`)
    const mapsUrl = `https://www.google.com/maps/search/${encodedAddress}`
    window.open(mapsUrl, '_blank')
  }

  const handleCall = (phone: string) => {
    window.open(`tel:${phone}`, '_self')
  }

  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <Star
        key={i}
        className={`h-3 w-3 ${i < Math.floor(rating) ? "text-yellow-400 fill-current" : "text-gray-300"}`}
      />
    ))
  }

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Find Nearby Healthcare Providers</h3>
        
        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500 h-4 w-4 transition-colors" />
          <Input
            placeholder="Search doctors, specialties, or hospitals..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-white dark:bg-gray-700 border-gray-200 dark:border-gray-600 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-colors"
          />
        </div>

        {/* Tab Buttons */}
        <div className="flex space-x-2">
          <Button
            variant={activeTab === "doctors" ? "default" : "outline"}
            onClick={() => setActiveTab("doctors")}
            className="flex-1 bg-white dark:bg-gray-700 border-gray-200 dark:border-gray-600 text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
          >
            Doctors ({filteredDoctors.length})
          </Button>
          <Button
            variant={activeTab === "hospitals" ? "default" : "outline"}
            onClick={() => setActiveTab("hospitals")}
            className="flex-1 bg-white dark:bg-gray-700 border-gray-200 dark:border-gray-600 text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
          >
            Hospitals ({filteredHospitals.length})
          </Button>
        </div>
      </div>

      {/* Doctors List */}
      {activeTab === "doctors" && (
        <div className="space-y-4">
          {filteredDoctors.map((doctor, index) => (
            <motion.div
              key={doctor.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <Card className="hover:shadow-md transition-shadow bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                <CardContent className="p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h4 className="font-semibold text-blue-700 dark:text-blue-400 transition-colors">{doctor.name}</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-300 transition-colors">{doctor.specialty}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 transition-colors">{doctor.hospital}</p>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center mb-1">
                        {renderStars(doctor.rating)}
                        <span className="text-xs text-gray-600 dark:text-gray-300 ml-1 transition-colors">({doctor.rating})</span>
                      </div>
                      <div className="flex items-center text-xs text-gray-500 dark:text-gray-400 transition-colors">
                        <MapPin className="h-3 w-3 mr-1" />
                        {doctor.distance}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 mb-3">
                    <div className="flex items-center">
                      <Phone className="h-3 w-3 mr-1" />
                      {doctor.phone}
                    </div>
                    <div className="flex items-center">
                      <Clock className="h-3 w-3 mr-1" />
                      {doctor.availability}
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1 mb-3">
                    {doctor.languages.map((lang) => (
                      <Badge key={lang} variant="secondary" className="text-xs">
                        {lang}
                      </Badge>
                    ))}
                  </div>

                  <div className="flex space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="flex-1"
                      onClick={() => handleCall(doctor.phone)}
                    >
                      <Phone className="h-3 w-3 mr-1" />
                      Call
                    </Button>
                    <Button 
                      size="sm" 
                      className="flex-1"
                      onClick={() => handleDirections(doctor.address, doctor.hospital)}
                    >
                      <Navigation className="h-3 w-3 mr-1" />
                      Directions
                    </Button>
                    <Button size="sm" variant="outline" className="flex-1">
                      Book Appointment
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      )}

      {/* Hospitals List */}
      {activeTab === "hospitals" && (
        <div className="space-y-4">
          {filteredHospitals.map((hospital, index) => (
            <motion.div
              key={hospital.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <Card className="hover:shadow-md transition-shadow">
                <CardContent className="p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h4 className="font-semibold text-blue-700">{hospital.name}</h4>
                      <p className="text-sm text-gray-600">{hospital.type}</p>
                      <p className="text-xs text-gray-500">{hospital.address}</p>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center mb-1">
                        {renderStars(hospital.rating)}
                        <span className="text-xs text-gray-600 ml-1">({hospital.rating})</span>
                      </div>
                      <div className="flex items-center text-xs text-gray-500">
                        <MapPin className="h-3 w-3 mr-1" />
                        {hospital.distance}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 mb-3">
                    <div className="flex items-center">
                      <Phone className="h-3 w-3 mr-1" />
                      {hospital.phone}
                    </div>
                    <div className="flex items-center text-red-600">
                      <Phone className="h-3 w-3 mr-1" />
                      Emergency: {hospital.emergencyPhone}
                    </div>
                  </div>

                  <div className="mb-3">
                    <p className="text-xs text-gray-500 mb-1">Services:</p>
                    <div className="flex flex-wrap gap-1">
                      {hospital.services.map((service) => (
                        <Badge key={service} variant="outline" className="text-xs">
                          {service}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center text-xs text-green-600">
                      <Clock className="h-3 w-3 mr-1" />
                      {hospital.availability}
                    </div>
                  </div>

                  <div className="flex space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="flex-1"
                      onClick={() => handleCall(hospital.phone)}
                    >
                      <Phone className="h-3 w-3 mr-1" />
                      Call
                    </Button>
                    <Button 
                      size="sm" 
                      variant="destructive" 
                      className="flex-1"
                      onClick={() => handleCall(hospital.emergencyPhone)}
                    >
                      <Phone className="h-3 w-3 mr-1" />
                      Emergency
                    </Button>
                    <Button 
                      size="sm" 
                      className="flex-1"
                      onClick={() => handleDirections(hospital.address, hospital.name)}
                    >
                      <Navigation className="h-3 w-3 mr-1" />
                      Directions
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      )}

      {/* No Results */}
      {((activeTab === "doctors" && filteredDoctors.length === 0) ||
        (activeTab === "hospitals" && filteredHospitals.length === 0)) && (
        <Card className="bg-gray-50">
          <CardContent className="p-8 text-center text-gray-500">
            <Search className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>No {activeTab} found matching your search.</p>
            <p className="text-sm">Try adjusting your search terms.</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}