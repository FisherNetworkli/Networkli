"use client"

import React from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import Head from 'next/head'
import Image from 'next/image'
import PublicPageWrapper from './components/PublicPageWrapper'

export default function Home() {
  return (
    <PublicPageWrapper>
      <Head>
        <title>Networkli - Professional Networking Reimagined for Introverts</title>
        <meta name="description" content="Connect with purpose, build meaningful professional relationships, and grow your career with Networkli. Our AI-powered platform helps introverts network comfortably and authentically." />
        <meta name="keywords" content="introvert networking, professional networking, AI matching, meaningful connections, career growth, networking app, introvert-friendly networking" />
      </Head>

      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center bg-connection-blue text-white overflow-hidden" aria-label="Hero section">
        <div className="absolute inset-0">
          <Image
            src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250413_1100_Social%20Media%20Interaction_remix_01jrr1d209e1ztp40mbsry0m56.png"
            alt="Professional networking at sunset"
            fill
            sizes="(max-width: 768px) 100vw, 100vw"
            quality={90}
            className="object-cover object-center md:hidden"
            priority
            style={{
              objectPosition: 'center 30%'
            }}
            onError={(e) => {
              console.error('Error loading hero image:', e);
              // Fallback to a solid color background if image fails to load
              e.currentTarget.style.backgroundColor = '#1e40af';
            }}
            placeholder="blur"
            blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDABQODxIPDRQSEBIXFRQdHx4eHRoaHSQtJSEkLzYvLy0vLi44QjY4OEI4Li8vQUVFRUVFRUVFRUVFRUVFRUVFRUX/2wBDAR0XFyAeIBogHiAeIBUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUX/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
          />
          <Image
            src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250413_1050_Social%20Gathering%20Animation_remix_01jrr0tkmeeaq9k1a9ez9qztyr.png"
            alt="Professional networking at sunset"
            fill
            sizes="(min-width: 769px) 100vw, 100vw"
            quality={90}
            className="object-cover object-center hidden md:block"
            priority
            style={{
              objectPosition: 'center 30%'
            }}
            onError={(e) => {
              console.error('Error loading hero image:', e);
              // Fallback to a solid color background if image fails to load
              e.currentTarget.style.backgroundColor = '#1e40af';
            }}
            placeholder="blur"
            blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDABQODxIPDRQSEBIXFRQdHx4eHRoaHSQtJSEkLzYvLy0vLi44QjY4OEI4Li8vQUVFRUVFRUVFRUVFRUVFRUVFRUX/2wBDAR0XFyAeIBogHiAeIBUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUX/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
          />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-connection-blue/50 to-connection-blue-70/80 z-10" />
        <motion.div 
          className="relative z-20 text-center px-4 max-w-4xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-6xl md:text-7xl font-bold mb-6 tracking-tight">
            Networkli
          </h1>
          <p className="text-xl md:text-2xl mb-4 text-gray-100">
            Professional networking reimagined for introverts and thoughtful professionals.
          </p>
          <p className="text-lg md:text-xl mb-8 text-gray-200">
            Build meaningful connections at your own pace, without the pressure of traditional networking.
          </p>
          <Link 
            href="/signup" 
            className="inline-block bg-white text-connection-blue px-8 py-4 rounded-full text-lg font-medium hover:bg-gray-100 transition-colors"
            aria-label="Get started with Networkli"
          >
            Get Started
          </Link>
        </motion.div>
      </section>

      {/* Feature 1: Introvert-Friendly Networking */}
      <section className="py-24 bg-white" aria-labelledby="introvert-networking-heading">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div 
              className="text-left"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 id="introvert-networking-heading" className="text-4xl md:text-5xl font-bold mb-6 text-gray-900">
                Networking Made Comfortable
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                No more awkward small talk or forced interactions. Our platform is designed for meaningful, purpose-driven connections that respect your communication style and energy levels.
              </p>
              <Link 
                href="/features/introvert-friendly" 
                className="text-connection-blue text-lg font-medium hover:text-connection-blue-70 transition-colors"
                aria-label="Learn more about our introvert-friendly approach"
              >
                Learn more about our introvert-friendly approach →
              </Link>
            </motion.div>
            <motion.div 
              className="relative rounded-2xl h-96 overflow-hidden"
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <div className="relative h-64 md:h-80 rounded-xl overflow-hidden">
                <Image
                  src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1519_Cozy%20Networking%20Lounge_simple_compose_01jrnxwpvafr2vv3b404hr8h2c.png"
                  alt="Networking Made Comfortable"
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 100vw, 50vw"
                />
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Feature 2: Smart Matching */}
      <section className="py-24 bg-gray-50" aria-labelledby="smart-matching-heading">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div 
              className="relative rounded-2xl h-96 overflow-hidden order-2 md:order-1"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <div className="relative h-64 md:h-80 rounded-xl overflow-hidden">
                <Image
                  src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1520_Perfect%20Match%20Found!_simple_compose_01jrnxy1f9ftkahscvm7fw5xj8.png"
                  alt="Smart Matching"
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 100vw, 50vw"
                />
              </div>
            </motion.div>
            <motion.div 
              className="text-left order-1 md:order-2"
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 id="smart-matching-heading" className="text-4xl md:text-5xl font-bold mb-6 text-gray-900">
                Smart Matching
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Our AI-powered algorithm connects you with like-minded professionals who share your interests and communication style, making conversations feel natural and comfortable.
              </p>
              <Link 
                href="/features/smart-matching" 
                className="text-connection-blue text-lg font-medium hover:text-connection-blue-70 transition-colors"
                aria-label="Learn more about Smart Matching"
              >
                Learn more about Smart Matching →
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Feature 3: Privacy First */}
      <section className="py-24 bg-white" aria-labelledby="privacy-heading">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div 
              className="text-left"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 id="privacy-heading" className="text-4xl md:text-5xl font-bold mb-6 text-gray-900">
                Privacy First
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Control your networking experience. Choose what to share, when to connect, and how to engage - all on your own terms.
              </p>
              <Link 
                href="/features/privacy" 
                className="text-connection-blue text-lg font-medium hover:text-connection-blue-70 transition-colors"
                aria-label="Learn more about our Privacy approach"
              >
                Learn more about our Privacy approach →
              </Link>
            </motion.div>
            <motion.div 
              className="relative rounded-2xl h-96 overflow-hidden"
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <div className="relative h-64 md:h-80 rounded-xl overflow-hidden">
                <Image
                  src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1521_Cozy%20Security%20Workspace_simple_compose_01jrnxz6raeagvgh3esfn2x3v0.png"
                  alt="Privacy First"
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 100vw, 50vw"
                />
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Feature 4: Meaningful Connections */}
      <section className="py-24 bg-gray-50" aria-labelledby="meaningful-connections-heading">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div 
              className="relative rounded-2xl h-96 overflow-hidden order-2 md:order-1"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <div className="relative h-64 md:h-80 rounded-xl overflow-hidden">
                <Image
                  src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1525_Cafe%20Connection%20Aligned_simple_compose_01jrny7phzeakvaw3xff9jn2aq.png"
                  alt="Meaningful Connections"
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 100vw, 50vw"
                />
              </div>
            </motion.div>
            <motion.div 
              className="text-left order-1 md:order-2"
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 id="meaningful-connections-heading" className="text-4xl md:text-5xl font-bold mb-6 text-gray-900">
                Meaningful Connections
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Focus on quality over quantity. Build genuine professional relationships through thoughtful interactions and shared interests.
              </p>
              <Link 
                href="/features/meaningful-connections" 
                className="text-connection-blue text-lg font-medium hover:text-connection-blue-70 transition-colors"
                aria-label="Learn more about our approach to meaningful connections"
              >
                Learn more about our approach to meaningful connections →
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-24 bg-white" aria-labelledby="testimonials-heading">
        <div className="max-w-7xl mx-auto px-4">
          <h2 id="testimonials-heading" className="text-4xl md:text-5xl font-bold mb-12 text-center">
            What Our Users Say
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <motion.div 
              className="bg-gray-50 p-8 rounded-xl shadow-sm"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-connection-blue rounded-full flex items-center justify-center text-white font-bold mr-4">
                  S
                </div>
                <div>
                  <h3 className="font-bold">Sarah M.</h3>
                  <p className="text-gray-600 text-sm">Software Engineer</p>
                </div>
              </div>
              <p className="text-gray-700">
                "As an introvert, I've always struggled with networking events. Networkli has completely changed how I connect with other professionals. The AI matching is spot-on, and I've made several meaningful connections without the usual anxiety."
              </p>
            </motion.div>
            <motion.div 
              className="bg-gray-50 p-8 rounded-xl shadow-sm"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-connection-blue rounded-full flex items-center justify-center text-white font-bold mr-4">
                  J
                </div>
                <div>
                  <h3 className="font-bold">James T.</h3>
                  <p className="text-gray-600 text-sm">Marketing Director</p>
                </div>
              </div>
              <p className="text-gray-700">
                "The conversation starters are brilliant. They help me break the ice naturally and focus on meaningful topics rather than awkward small talk. I've expanded my professional network significantly since joining Networkli."
              </p>
            </motion.div>
            <motion.div 
              className="bg-gray-50 p-8 rounded-xl shadow-sm"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              viewport={{ once: true }}
            >
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-connection-blue rounded-full flex items-center justify-center text-white font-bold mr-4">
                  A
                </div>
                <div>
                  <h3 className="font-bold">Alex K.</h3>
                  <p className="text-gray-600 text-sm">Freelance Designer</p>
                </div>
              </div>
              <p className="text-gray-700">
                "I love how I can control my networking pace. The platform respects my energy levels and helps me connect with people who understand introverted communication styles. It's been a game-changer for my career."
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-connection-blue text-white" aria-labelledby="cta-heading">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 id="cta-heading" className="text-4xl md:text-5xl font-bold mb-8">
              Network authentically, at your own pace
            </h2>
            <p className="text-xl text-gray-100 mb-12">
              Join a community of professionals who value meaningful connections over superficial networking.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link 
                href="/signup" 
                className="inline-block bg-white text-connection-blue px-8 py-4 rounded-full text-lg font-medium hover:bg-gray-100 transition-colors"
                aria-label="Start your journey with Networkli"
              >
                Start Your Journey
              </Link>
              <Link 
                href="/learn-more" 
                className="inline-block bg-transparent text-white px-8 py-4 rounded-full text-lg font-medium border border-white hover:bg-white/10 transition-colors"
                aria-label="Learn more about Networkli"
              >
                Learn More
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 bg-white" aria-labelledby="faq-heading">
        <div className="max-w-4xl mx-auto px-4">
          <h2 id="faq-heading" className="text-4xl md:text-5xl font-bold mb-12 text-center">
            Frequently Asked Questions
          </h2>
          <div className="space-y-8">
            <motion.div 
              className="bg-gray-50 p-8 rounded-xl"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h3 className="text-2xl font-bold mb-4">How does Networkli help introverts network?</h3>
              <p className="text-gray-700">
                Networkli is specifically designed for introverts and thoughtful professionals. Our AI matching algorithm connects you with like-minded individuals who share your communication style and interests. You can network at your own pace, without the pressure of traditional networking events, and our conversation starters help break the ice naturally.
              </p>
            </motion.div>
            <motion.div 
              className="bg-gray-50 p-8 rounded-xl"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <h3 className="text-2xl font-bold mb-4">How does the AI matching algorithm work?</h3>
              <p className="text-gray-700">
                Our patent-pending AI algorithm analyzes multiple factors including personality traits, communication preferences, professional interests, and career goals. It then matches you with professionals who complement your networking style and share meaningful commonalities, making conversations feel natural and productive.
              </p>
            </motion.div>
            <motion.div 
              className="bg-gray-50 p-8 rounded-xl"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              viewport={{ once: true }}
            >
              <h3 className="text-2xl font-bold mb-4">Can I control my privacy settings?</h3>
              <p className="text-gray-700">
                Absolutely. Networkli puts you in control of your networking experience. You can choose what information to share, when to connect with others, and how to engage with your network. Our privacy-first approach ensures you're comfortable with every interaction.
              </p>
            </motion.div>
          </div>
        </div>
      </section>
    </PublicPageWrapper>
  )
} 