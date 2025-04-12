"use client"

import React from 'react'
import { motion } from 'framer-motion'

export default function AccessibilityPage() {
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
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Accessibility Statement</h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Our commitment to digital accessibility
            </p>
          </motion.div>
        </div>
      </section>

      {/* Content Section */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4">
          <div className="prose prose-lg max-w-none">
            <h2>1. Our Commitment</h2>
            <p>
              Networkli is committed to ensuring digital accessibility for people with disabilities. 
              We are continually improving the user experience for everyone and applying the relevant 
              accessibility standards.
            </p>

            <h2>2. Conformance Status</h2>
            <p>
              The Web Content Accessibility Guidelines (WCAG) defines requirements for designers and 
              developers to improve accessibility for people with disabilities. It defines three levels 
              of conformance: Level A, Level AA, and Level AAA.
            </p>
            <p>
              Networkli is partially conformant with WCAG 2.1 level AA. Partially conformant means 
              that some parts of the content do not fully conform to the accessibility standard.
            </p>

            <h2>3. Accessibility Features</h2>
            <p>Networkli includes the following accessibility features:</p>
            <ul>
              <li>Keyboard navigation support</li>
              <li>Text alternatives for non-text content</li>
              <li>Captions and other alternatives for multimedia</li>
              <li>Content can be presented in different ways</li>
              <li>Users have enough time to read and use the content</li>
              <li>Content does not cause seizures or physical reactions</li>
              <li>Users can easily navigate, find content, and determine where they are</li>
              <li>Text is readable and understandable</li>
              <li>Content appears and operates in predictable ways</li>
              <li>Users are helped to avoid and correct mistakes</li>
              <li>Content is compatible with current and future user tools</li>
            </ul>

            <h2>4. Compatibility with Browsers and Assistive Technology</h2>
            <p>
              Networkli is designed to be compatible with the following assistive technologies:
            </p>
            <ul>
              <li>Screen readers (NVDA, VoiceOver, TalkBack)</li>
              <li>Screen magnification software</li>
              <li>Speech recognition software</li>
              <li>Alternative keyboard and mouse input devices</li>
            </ul>

            <h2>5. Technical Specifications</h2>
            <p>
              Accessibility of Networkli relies on the following technologies to work with the 
              particular combination of web browser and any assistive technologies or plugins 
              installed on your computer:
            </p>
            <ul>
              <li>HTML</li>
              <li>WAI-ARIA</li>
              <li>CSS</li>
              <li>JavaScript</li>
            </ul>

            <h2>6. Limitations and Alternatives</h2>
            <p>
              Despite our best efforts to ensure accessibility of Networkli, there may be some 
              limitations. Below is a description of known limitations, and potential solutions. 
              Please contact us if you observe an issue not listed below.
            </p>

            <h2>7. Assessment Approach</h2>
            <p>
              Networkli assessed the accessibility of this website by the following approaches:
            </p>
            <ul>
              <li>Self-evaluation</li>
              <li>External evaluation</li>
              <li>User testing</li>
            </ul>

            <h2>8. Feedback</h2>
            <p>
              We welcome your feedback on the accessibility of Networkli. Please let us know if you 
              encounter accessibility barriers on Networkli:
            </p>
            <ul>
              <li>Email: accessibility@networkli.com</li>
              <li>Phone: (555) 123-4567</li>
              <li>Postal address: 123 Main Street, Suite 100, San Francisco, CA 94105</li>
            </ul>
            <p>
              We try to respond to feedback within 5 business days.
            </p>

            <h2>9. Date</h2>
            <p>
              This statement was created on {new Date().toLocaleDateString()}.
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