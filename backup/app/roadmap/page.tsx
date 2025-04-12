"use client"

import React from 'react'
import { motion } from 'framer-motion'

export default function RoadmapPage() {
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
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Product Roadmap</h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Our vision for the future of professional networking
            </p>
          </motion.div>
        </div>
      </section>

      {/* Content Section */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4">
          <div className="prose prose-lg max-w-none">
            <h2>Current Focus (Q2 2024)</h2>
            <ul>
              <li>
                <strong>Enhanced AI Matching</strong>
                <ul>
                  <li>Improved personality compatibility algorithm</li>
                  <li>Better conversation starter suggestions</li>
                  <li>More accurate skill matching</li>
                </ul>
              </li>
              <li>
                <strong>Mobile App Improvements</strong>
                <ul>
                  <li>Offline mode support</li>
                  <li>Enhanced push notifications</li>
                  <li>Improved performance and stability</li>
                </ul>
              </li>
              <li>
                <strong>Privacy Features</strong>
                <ul>
                  <li>Advanced privacy controls</li>
                  <li>Data export functionality</li>
                  <li>Enhanced security measures</li>
                </ul>
              </li>
            </ul>

            <h2>Coming Soon (Q3 2024)</h2>
            <ul>
              <li>
                <strong>Event Integration</strong>
                <ul>
                  <li>Virtual event platform</li>
                  <li>Event-specific networking features</li>
                  <li>Automated follow-up suggestions</li>
                </ul>
              </li>
              <li>
                <strong>Enterprise Features</strong>
                <ul>
                  <li>Team networking spaces</li>
                  <li>Admin dashboard</li>
                  <li>Custom branding options</li>
                </ul>
              </li>
              <li>
                <strong>Learning Resources</strong>
                <ul>
                  <li>Networking tips and guides</li>
                  <li>Professional development content</li>
                  <li>Community best practices</li>
                </ul>
              </li>
            </ul>

            <h2>Future Vision (Q4 2024)</h2>
            <ul>
              <li>
                <strong>Advanced Analytics</strong>
                <ul>
                  <li>Network growth insights</li>
                  <li>Connection quality metrics</li>
                  <li>Personalized recommendations</li>
                </ul>
              </li>
              <li>
                <strong>Integration Expansion</strong>
                <ul>
                  <li>Calendar integration</li>
                  <li>CRM system connections</li>
                  <li>Professional social networks</li>
                </ul>
              </li>
              <li>
                <strong>Community Features</strong>
                <ul>
                  <li>Interest-based groups</li>
                  <li>Knowledge sharing platform</li>
                  <li>Mentorship matching</li>
                </ul>
              </li>
            </ul>

            <h2>Long-term Goals (2025)</h2>
            <ul>
              <li>
                <strong>Global Expansion</strong>
                <ul>
                  <li>Multi-language support</li>
                  <li>Regional networking features</li>
                  <li>Cultural adaptation</li>
                </ul>
              </li>
              <li>
                <strong>AI Advancements</strong>
                <ul>
                  <li>Predictive networking</li>
                  <li>Career path suggestions</li>
                  <li>Automated relationship management</li>
                </ul>
              </li>
              <li>
                <strong>Platform Evolution</strong>
                <ul>
                  <li>AR/VR networking experiences</li>
                  <li>Blockchain integration</li>
                  <li>Advanced privacy features</li>
                </ul>
              </li>
            </ul>

            <h2>Feedback & Suggestions</h2>
            <p>
              We value your input in shaping our product roadmap. If you have suggestions for features 
              or improvements, please reach out to us at:
              <br />
              Email: product@networkli.com
            </p>

            <p className="text-sm text-gray-500 mt-8">
              Last updated: {new Date().toLocaleDateString()}
            </p>
          </div>
        </div>
      </section>
    </div>
  )
} 