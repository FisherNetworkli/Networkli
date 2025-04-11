"use client"

import React from 'react'
import { motion } from 'framer-motion'

export default function SecurityPage() {
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
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Security</h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Your data security is our top priority
            </p>
          </motion.div>
        </div>
      </section>

      {/* Content Section */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4">
          <div className="prose prose-lg max-w-none">
            <h2>1. Data Protection</h2>
            <p>
              At Networkli, we implement industry-standard security measures to protect your data:
            </p>
            <ul>
              <li>End-to-end encryption for all communications</li>
              <li>Secure data storage with regular backups</li>
              <li>Regular security audits and penetration testing</li>
              <li>Compliance with data protection regulations</li>
            </ul>

            <h2>2. Authentication & Access Control</h2>
            <p>
              We maintain strict access controls to protect your account:
            </p>
            <ul>
              <li>Multi-factor authentication (MFA) support</li>
              <li>Secure password hashing and storage</li>
              <li>Role-based access control</li>
              <li>Regular security training for our team</li>
            </ul>

            <h2>3. Infrastructure Security</h2>
            <p>
              Our infrastructure is built with security in mind:
            </p>
            <ul>
              <li>Cloud infrastructure with enterprise-grade security</li>
              <li>DDoS protection and mitigation</li>
              <li>Regular security updates and patches</li>
              <li>24/7 monitoring and incident response</li>
            </ul>

            <h2>4. Privacy Controls</h2>
            <p>
              You have full control over your data:
            </p>
            <ul>
              <li>Granular privacy settings</li>
              <li>Data export capabilities</li>
              <li>Account deletion options</li>
              <li>Transparent data usage policies</li>
            </ul>

            <h2>5. Compliance</h2>
            <p>
              We maintain compliance with relevant regulations:
            </p>
            <ul>
              <li>GDPR compliance</li>
              <li>CCPA compliance</li>
              <li>Industry-standard security certifications</li>
              <li>Regular compliance audits</li>
            </ul>

            <h2>6. Incident Response</h2>
            <p>
              Our security team is ready to respond to any security incidents:
            </p>
            <ul>
              <li>24/7 security monitoring</li>
              <li>Rapid incident response</li>
              <li>Transparent communication</li>
              <li>Regular security updates</li>
            </ul>

            <h2>7. Reporting Security Issues</h2>
            <p>
              If you discover a security vulnerability, please report it to:
              <br />
              Email: security@networkli.com
            </p>
            <p>
              We take all security reports seriously and will respond promptly.
            </p>

            <h2>8. Security Updates</h2>
            <p>
              We regularly update our security measures and will notify users of any significant changes 
              that may affect their data or privacy.
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