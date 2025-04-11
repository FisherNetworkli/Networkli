"use client"

import React from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'

export default function DownloadPage() {
  const platforms = [
    {
      id: 'ios',
      name: 'iOS App',
      icon: '📱',
      description: 'Download for iPhone and iPad',
      link: '#',
      features: ['Full mobile experience', 'Push notifications', 'Offline mode', 'Biometric login']
    },
    {
      id: 'android',
      name: 'Android App',
      icon: '🤖',
      description: 'Download for Android devices',
      link: '#',
      features: ['Full mobile experience', 'Push notifications', 'Offline mode', 'Biometric login']
    },
    {
      id: 'desktop',
      name: 'Desktop App',
      icon: '💻',
      description: 'Download for Windows and Mac',
      link: '#',
      features: ['Native desktop experience', 'System notifications', 'Quick actions', 'Keyboard shortcuts']
    }
  ]

  const features = [
    {
      title: 'Cross-Platform Sync',
      description: 'Seamlessly sync your data across all your devices'
    },
    {
      title: 'Offline Access',
      description: 'Access your network and messages even without internet'
    },
    {
      title: 'Push Notifications',
      description: 'Stay updated with real-time notifications'
    },
    {
      title: 'Enhanced Security',
      description: 'Biometric authentication and end-to-end encryption'
    }
  ]

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
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Download Networkli</h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Get the best networking experience on your favorite devices
            </p>
          </motion.div>
        </div>
      </section>

      {/* Download Options */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-8">
            {platforms.map((platform, index) => (
              <motion.div
                key={platform.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className="text-4xl mb-4">{platform.icon}</div>
                <h2 className="text-2xl font-bold mb-2">{platform.name}</h2>
                <p className="text-gray-600 mb-6">{platform.description}</p>
                <ul className="space-y-2 mb-6">
                  {platform.features.map((feature, i) => (
                    <li key={i} className="flex items-center text-gray-600">
                      <span className="mr-2">✓</span>
                      {feature}
                    </li>
                  ))}
                </ul>
                <Link
                  href={platform.link}
                  className="block w-full text-center px-6 py-3 rounded-full text-white bg-connection-blue hover:bg-connection-blue-70 transition-colors"
                >
                  Download Now
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-4xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold mb-4">App Features</h2>
            <p className="text-gray-600">
              Experience the full power of Networkli on your devices
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm"
              >
                <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* System Requirements */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold mb-4">System Requirements</h2>
            <p className="text-gray-600">
              Make sure your device meets the minimum requirements
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                platform: 'iOS',
                requirements: ['iOS 14.0 or later', 'iPhone or iPad', '2GB RAM minimum']
              },
              {
                platform: 'Android',
                requirements: ['Android 8.0 or later', '2GB RAM minimum', '100MB free storage']
              },
              {
                platform: 'Desktop',
                requirements: ['Windows 10/11 or macOS 10.15+', '4GB RAM minimum', '500MB free storage']
              }
            ].map((os, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-gray-50 p-6 rounded-lg"
              >
                <h3 className="text-xl font-bold mb-4">{os.platform}</h3>
                <ul className="space-y-2">
                  {os.requirements.map((req, i) => (
                    <li key={i} className="flex items-center text-gray-600">
                      <span className="mr-2">•</span>
                      {req}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
} 