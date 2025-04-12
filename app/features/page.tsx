"use client"

import React from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'

const features = [
  {
    title: "AI-Powered Matchmaking",
    description: "Our intelligent algorithm finds the perfect professional connections based on your interests, goals, and communication style.",
    image: "https://networkly.supabase.co/storage/v1/object/public/images/features/20250412_1530_Magical Matchmaking Moment_simple_compose_01jrnyf9nhe00tqq8s5kmtqyzd.png"
  },
  {
    title: "Conversation Starters",
    description: "Never worry about how to break the ice. Our platform provides personalized conversation starters based on shared interests.",
    image: "https://networkly.supabase.co/storage/v1/object/public/images/features/20250412_1530_Joyful Virtual Connection_simple_compose_01jrnygj5cerqbk7k4zs2x4a52.png"
  },
  {
    title: "Event Integration",
    description: "Seamlessly integrate with professional events and conferences to make meaningful connections in person.",
    image: "https://networkly.supabase.co/storage/v1/object/public/images/features/20250412_1532_Event Dashboard Magic_simple_compose_01jrnykb4kfevs7day7t71z5w0.png"
  },
  {
    title: "Connection Analytics",
    description: "Track your networking progress and gain insights into your professional relationships.",
    image: "https://networkly.supabase.co/storage/v1/object/public/images/features/20250412_1533_Joyful Data Review_simple_compose_01jrnypd2we5ztk3v5069bbs18.png"
  },
  {
    title: "Enterprise API Access",
    description: "Integrate Networkli's powerful networking features into your own platform with our enterprise API.",
    image: "https://networkly.supabase.co/storage/v1/object/public/images/features/20250412_1536_Tech Partnership Celebration_simple_compose_01jrnnyvbbdfwnvzzvwp8n5wzbx.png"
  }
]

export default function FeaturesPage() {
  return (
    <div className="min-h-screen bg-white">
      {/* Hero Section */}
      <section className="relative h-[50vh] min-h-[400px]">
        <Image
          src="https://networkly.supabase.co/storage/v1/object/public/images/features/20250412_1519_Cozy Networking Lounge_simple_compose_01jrnxwpvafr2vv3b404hr8h2c.png"
          alt="Feature highlights"
          fill
          className="object-cover"
          sizes="100vw"
          priority
        />
        <div className="absolute inset-0 bg-black/40" />
        <div className="absolute inset-0 flex items-center">
          <div className="container mx-auto px-4">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Features
            </h1>
            <p className="text-xl text-white/90 max-w-2xl">
              Discover how Networkli makes professional networking comfortable and meaningful.
            </p>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-24">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                className="bg-white rounded-xl overflow-hidden shadow-lg"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className="relative h-64">
                  <Image
                    src={feature.image}
                    alt={feature.title}
                    fill
                    className="object-cover"
                    sizes="(max-width: 768px) 100vw, 50vw"
                  />
                </div>
                <div className="p-6">
                  <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                  <p className="text-gray-600">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
} 