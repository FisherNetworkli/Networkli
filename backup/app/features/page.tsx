"use client"

import React from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'

export default function FeaturesPage() {
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
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Platform Features</h1>
            <p className="text-xl text-gray-100 max-w-3xl mx-auto">
              Discover how our innovative features make networking more meaningful and comfortable
            </p>
          </motion.div>
        </div>
      </section>

      {/* AI Matching Algorithm */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="text-left"
            >
              <h2 className="text-4xl font-bold mb-6">AI-Powered Matching</h2>
              <p className="text-xl text-gray-600 mb-8">
                Our patent-pending AI algorithm analyzes multiple factors including personality, communication style, skills, and professional goals to create meaningful connections that matter.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Deep learning algorithm for precise compatibility matching</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Personality-based networking recommendations</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Skills and interests alignment for meaningful conversations</span>
                </li>
              </ul>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="relative h-96 rounded-2xl overflow-hidden"
            >
              <Image
                src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1530_Magical%20Matchmaking%20Moment_simple_compose_01jrnyf9nhe00tqq8s5kmtqyzd.png"
                alt="AI-Powered Matching"
                fill
                className="object-cover"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Custom Conversation Starters */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="relative h-96 rounded-2xl overflow-hidden order-2 md:order-1"
            >
              <Image
                src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1530_Joyful%20Virtual%20Connection_simple_compose_01jrnygj5cerqbk7k4zs2x4a52.png"
                alt="Conversation Starters"
                fill
                className="object-cover"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="text-left order-1 md:order-2"
            >
              <h2 className="text-4xl font-bold mb-6">Smart Conversation Starters</h2>
              <p className="text-xl text-gray-600 mb-8">
                Break the ice naturally with AI-generated conversation starters based on shared interests and professional backgrounds.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Personalized conversation prompts</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Industry-specific discussion topics</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Common interest highlighting</span>
                </li>
              </ul>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Event Integration */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="text-left"
            >
              <h2 className="text-4xl font-bold mb-6">Event Integration</h2>
              <p className="text-xl text-gray-600 mb-8">
                Transform your events with our AI-driven compatibility matching for attendees, making networking more effective and enjoyable.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Pre-event matching for optimal connections</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Real-time networking suggestions</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Post-event connection nurturing</span>
                </li>
              </ul>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="relative h-96 rounded-2xl overflow-hidden"
            >
              <Image
                src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1532_Event%20Dashboard%20Magic_simple_compose_01jrnykb4kfevs7day7t71z5w0.png"
                alt="Event Integration"
                fill
                className="object-cover"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Analytics Dashboard */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="relative h-96 rounded-2xl overflow-hidden order-2 md:order-1"
            >
              <Image
                src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1533_Joyful%20Data%20Review_simple_compose_01jrnypd2we5ztk3v5069bbs18.png"
                alt="Connection Analytics"
                fill
                className="object-cover"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="text-left order-1 md:order-2"
            >
              <h2 className="text-4xl font-bold mb-6">Connection Analytics</h2>
              <p className="text-xl text-gray-600 mb-8">
                Track and optimize your networking effectiveness with detailed analytics and insights.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Connection quality metrics</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Engagement level tracking</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Network growth insights</span>
                </li>
              </ul>
            </motion.div>
          </div>
        </div>
      </section>

      {/* API Integration */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="text-left"
            >
              <h2 className="text-4xl font-bold mb-6">Enterprise API Access</h2>
              <p className="text-xl text-gray-600 mb-8">
                Integrate our powerful AI matching algorithm into your own platforms and applications.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>RESTful API integration</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Custom implementation support</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">•</span>
                  <span>Scalable solutions for any size organization</span>
                </li>
              </ul>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="relative h-96 rounded-2xl overflow-hidden"
            >
              <Image
                src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images/20250412_1536_Tech%20Partnership%20Celebration_simple_compose_01jrnyvbbdfwnvzzvwp8n5wzbx.png"
                alt="Enterprise API Access"
                fill
                className="object-cover"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  )
} 