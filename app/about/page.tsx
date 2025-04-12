"use client"

import React from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'
import PublicPageWrapper from '../components/PublicPageWrapper'

export default function AboutPage() {
  return (
    <PublicPageWrapper>
      <div className="min-h-screen bg-white">
        {/* Hero Section */}
        <section className="relative h-[60vh] min-h-[400px]">
          <Image
            src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1352_Team%20Spirit%20in%20Orange_simple_compose_01jrnrvkecf9h8jb72pf1zy3p5.png"
            alt="Our Team"
            fill
            className="object-cover"
            sizes="(max-width: 768px) 100vw, 50vw"
          />
          <div className="absolute inset-0 bg-black/40" />
          <div className="absolute inset-0 flex items-center">
            <div className="container mx-auto px-4">
              <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
                About Networkli
              </h1>
              <p className="text-xl text-white/90 max-w-2xl">
                Building meaningful professional connections for introverts and thoughtful networkers.
              </p>
            </div>
          </div>
        </section>

        {/* Origin Story */}
        <section className="py-16">
          <div className="max-w-7xl mx-auto px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="prose prose-lg mx-auto"
            >
              <h2 className="text-3xl font-bold mb-6">The Birth of Networkli</h2>
              <p className="text-gray-600 mb-8">
                Networkli was created by Dan and Brittany Fisher, combining their unique expertise and shared vision. Brittany's background as a speech-language pathologist and passion for neuroscience perfectly complemented Dan's experience as a serial entrepreneur with expertise in sales and IT. After successfully scaling a legal marketing business together, they identified a crucial gap in professional networking and decided to create a platform that would make meaningful connections possible for everyone, especially those who find traditional networking challenging.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Team Section */}
        <section className="py-16 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">Meet Our Team</h2>
            <div className="grid md:grid-cols-3 gap-8">
              {/* Dan Fisher */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm"
              >
                <h3 className="text-xl font-bold mb-2">Dan Fisher</h3>
                <p className="text-networkli-orange font-medium mb-4">CEO & Co-Founder</p>
                <p className="text-gray-600">
                  Serial entrepreneur with a background in sales and IT. Former Apple employee and creator of multiple 6-figure businesses, bringing technical expertise and business acumen to Networkli.
                </p>
              </motion.div>

              {/* Brittany Fisher */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm"
              >
                <h3 className="text-xl font-bold mb-2">Brittany Furnari Fisher</h3>
                <p className="text-networkli-orange font-medium mb-4">COO & Co-Founder</p>
                <p className="text-gray-600">
                  With a background in speech-language pathology and a passion for neuroscience, Brittany brings expertise in communication and human connection to help users build meaningful professional relationships.
                </p>
              </motion.div>

              {/* Shubham Chandra */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm"
              >
                <h3 className="text-xl font-bold mb-2">Shubham Chandra</h3>
                <p className="text-networkli-orange font-medium mb-4">CIO</p>
                <p className="text-gray-600">
                  Leading our technology development and strategic partnerships, Shubham oversees licensing and the development of our patent-pending AI matching algorithm.
                </p>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Achievements Section */}
        <section className="py-16">
          <div className="max-w-7xl mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">Our Achievements</h2>
            <div className="grid md:grid-cols-2 gap-8">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm border border-gray-100"
              >
                <h3 className="text-xl font-bold mb-4">Business Success</h3>
                <p className="text-gray-600">
                  Successfully scaled a legal marketing business to over $400K in revenue within just six months, demonstrating our ability to grow and scale effectively.
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm border border-gray-100"
              >
                <h3 className="text-xl font-bold mb-4">Innovation in Technology</h3>
                <p className="text-gray-600">
                  Developed a patent-pending AI matching algorithm that revolutionizes how professionals connect, making networking more meaningful and effective.
                </p>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Values Section */}
        <section className="py-16 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">Our Core Values</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <h3 className="text-xl font-bold mb-4">Meaningful Connections</h3>
                <p className="text-gray-600">
                  We believe in quality over quantity, fostering genuine professional relationships that last.
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <h3 className="text-xl font-bold mb-4">Inclusive Networking</h3>
                <p className="text-gray-600">
                  Creating a platform where everyone, especially introverts, can network comfortably and authentically.
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <h3 className="text-xl font-bold mb-4">Innovation</h3>
                <p className="text-gray-600">
                  Leveraging cutting-edge AI technology to transform how professionals connect and grow their networks.
                </p>
              </motion.div>
            </div>
          </div>
        </section>
      </div>
    </PublicPageWrapper>
  )
} 