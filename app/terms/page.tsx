"use client"

import React from 'react'
import { motion } from 'framer-motion'

export default function TermsPage() {
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
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Terms of Service</h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Please read these terms carefully before using our platform
            </p>
          </motion.div>
        </div>
      </section>

      {/* Content Section */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4">
          <div className="prose prose-lg max-w-none">
            <h2>1. Agreement to Terms</h2>
            <p>
              By accessing or using Networkli, you agree to be bound by these Terms of Service. 
              If you disagree with any part of the terms, you may not access the platform.
            </p>

            <h2>2. Use License</h2>
            <p>
              Permission is granted to temporarily access Networkli for personal, non-commercial 
              networking purposes, subject to these Terms of Service.
            </p>

            <h2>3. User Responsibilities</h2>
            <p>As a user of Networkli, you agree to:</p>
            <ul>
              <li>Provide accurate and complete information</li>
              <li>Maintain the security of your account</li>
              <li>Use the platform in compliance with all applicable laws</li>
              <li>Respect the privacy and rights of other users</li>
            </ul>

            <h2>4. Prohibited Activities</h2>
            <p>Users may not:</p>
            <ul>
              <li>Use the platform for any illegal purposes</li>
              <li>Harass, abuse, or harm other users</li>
              <li>Share false or misleading information</li>
              <li>Attempt to access unauthorized areas of the platform</li>
              <li>Use automated methods to access or interact with the platform</li>
            </ul>

            <h2>5. Content Guidelines</h2>
            <p>
              All content posted on Networkli must be:
            </p>
            <ul>
              <li>Accurate and truthful</li>
              <li>Respectful and professional</li>
              <li>Compliant with intellectual property rights</li>
              <li>Appropriate for a professional networking platform</li>
            </ul>

            <h2>6. Termination</h2>
            <p>
              We reserve the right to terminate or suspend access to our platform immediately, 
              without prior notice, for any breach of these Terms of Service.
            </p>

            <h2>7. Disclaimer</h2>
            <p>
              Networkli is provided "as is" without any warranties, expressed or implied. 
              We do not guarantee that the platform will be error-free or uninterrupted.
            </p>

            <h2>8. Limitation of Liability</h2>
            <p>
              Networkli shall not be liable for any indirect, incidental, special, consequential, 
              or punitive damages resulting from your use or inability to use the platform.
            </p>

            <h2>9. Changes to Terms</h2>
            <p>
              We reserve the right to modify these terms at any time. We will notify users of 
              any material changes by posting the new Terms of Service on this page.
            </p>

            <h2>10. Contact Information</h2>
            <p>
              If you have any questions about these Terms of Service, please contact us at:
              <br />
              Email: legal@networkli.com
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