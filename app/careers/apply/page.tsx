"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'

export default function ApplyPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    linkedin: '',
    github: '',
    portfolio: '',
    experience: '',
    availability: '',
    salary: '',
    referral: '',
    videoUrl: ''
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    
    // Create email body with form data
    const emailBody = `
Name: ${formData.name}
Email: ${formData.email}
Phone: ${formData.phone}
LinkedIn: ${formData.linkedin}
GitHub: ${formData.github}
Portfolio: ${formData.portfolio}
Experience: ${formData.experience}
Availability: ${formData.availability}
Salary Expectations: ${formData.salary}
Referral: ${formData.referral}
Video URL: ${formData.videoUrl}
    `
    
    // Open email client with pre-filled data
    window.location.href = `mailto:dan@networkly.ai?subject=Application for Founding Engineer Position&body=${encodeURIComponent(emailBody)}`
  }

  return (
    <div className="bg-white">
      {/* Hero Section */}
      <section className="pt-24 pb-12 bg-connection-blue text-white">
        <div className="max-w-7xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Apply for Founding Engineer</h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Join our team and help build the future of professional networking
            </p>
          </motion.div>
        </div>
      </section>

      {/* Application Form */}
      <section className="py-16">
        <div className="max-w-3xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-white p-8 rounded-lg shadow-md"
          >
            <h2 className="text-2xl font-bold mb-6">Application Form</h2>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">Full Name *</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                  />
                </div>
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">Email Address *</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                  />
                </div>
                <div>
                  <label htmlFor="phone" className="block text-sm font-medium text-gray-700 mb-1">Phone Number</label>
                  <input
                    type="tel"
                    id="phone"
                    name="phone"
                    value={formData.phone}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                  />
                </div>
                <div>
                  <label htmlFor="linkedin" className="block text-sm font-medium text-gray-700 mb-1">LinkedIn Profile</label>
                  <input
                    type="url"
                    id="linkedin"
                    name="linkedin"
                    value={formData.linkedin}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                  />
                </div>
                <div>
                  <label htmlFor="github" className="block text-sm font-medium text-gray-700 mb-1">GitHub Profile</label>
                  <input
                    type="url"
                    id="github"
                    name="github"
                    value={formData.github}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                  />
                </div>
                <div>
                  <label htmlFor="portfolio" className="block text-sm font-medium text-gray-700 mb-1">Portfolio Website</label>
                  <input
                    type="url"
                    id="portfolio"
                    name="portfolio"
                    value={formData.portfolio}
                    onChange={handleChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="experience" className="block text-sm font-medium text-gray-700 mb-1">Years of Experience *</label>
                <input
                  type="text"
                  id="experience"
                  name="experience"
                  value={formData.experience}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                />
              </div>

              <div>
                <label htmlFor="availability" className="block text-sm font-medium text-gray-700 mb-1">Availability *</label>
                <select
                  id="availability"
                  name="availability"
                  value={formData.availability}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                >
                  <option value="">Select availability</option>
                  <option value="Immediate">Immediate</option>
                  <option value="2 weeks">2 weeks</option>
                  <option value="1 month">1 month</option>
                  <option value="2 months">2 months</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div>
                <label htmlFor="salary" className="block text-sm font-medium text-gray-700 mb-1">Salary Expectations</label>
                <input
                  type="text"
                  id="salary"
                  name="salary"
                  value={formData.salary}
                  onChange={handleChange}
                  placeholder="e.g. $120,000 - $150,000"
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                />
              </div>

              <div>
                <label htmlFor="referral" className="block text-sm font-medium text-gray-700 mb-1">How did you hear about us?</label>
                <input
                  type="text"
                  id="referral"
                  name="referral"
                  value={formData.referral}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                />
              </div>

              <div>
                <label htmlFor="videoUrl" className="block text-sm font-medium text-gray-700 mb-1">Video URL *</label>
                <input
                  type="url"
                  id="videoUrl"
                  name="videoUrl"
                  value={formData.videoUrl}
                  onChange={handleChange}
                  required
                  placeholder="Link to your video (YouTube, Vimeo, etc.)"
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-connection-blue focus:border-connection-blue"
                />
              </div>

              <div className="bg-gray-50 p-4 rounded-md">
                <h3 className="text-lg font-medium mb-2">Video Instructions</h3>
                <p className="text-gray-600 mb-4">
                  Please record a short video (2-5 minutes) addressing the following points:
                </p>
                <ul className="list-disc pl-5 space-y-2 text-gray-600">
                  <li>A quick overview of your tech leadership experience</li>
                  <li>What's most meaningful to you right now, personally or professionally</li>
                  <li>Your thoughts on the loneliness epidemic and how tech might address it</li>
                  <li>How you envision impacting Networkli as a founding developer</li>
                </ul>
                <p className="text-gray-600 mt-4">
                  Don't worry about production quality – we're more interested in your ideas and passion. Just be your authentic self.
                </p>
              </div>

              <div className="pt-4">
                <button
                  type="submit"
                  className="w-full bg-connection-blue text-white py-3 px-6 rounded-md hover:bg-connection-blue-70 transition-colors"
                >
                  Submit Application
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      </section>
    </div>
  )
} 