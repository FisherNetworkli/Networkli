"use client"

import React from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'

export default function BlogPage() {
  const featuredPost = {
    title: "Introducing Networkli's AI-Powered Matching Algorithm",
    excerpt: "Discover how our advanced AI technology helps introverts make meaningful professional connections.",
    date: "March 15, 2024",
    author: "Shubham Chandra",
    category: "Technology",
    image: "/blog/ai-matching.jpg"
  }

  const recentPosts = [
    {
      title: "The Future of Professional Networking",
      excerpt: "How AI is transforming the way we connect professionally.",
      date: "March 10, 2024",
      author: "Brittany Furnari Fisher",
      category: "Industry Insights"
    },
    {
      title: "Building a More Inclusive Networking Platform",
      excerpt: "Our commitment to making professional networking accessible to everyone.",
      date: "March 5, 2024",
      author: "Dan",
      category: "Company News"
    },
    {
      title: "Tips for Introverts: Making the Most of Networkli",
      excerpt: "Learn how to leverage our platform for meaningful connections.",
      date: "March 1, 2024",
      author: "Brittany Furnari Fisher",
      category: "Tips & Tricks"
    }
  ]

  const categories = [
    "All Posts",
    "Company News",
    "Technology",
    "Industry Insights",
    "Tips & Tricks",
    "Success Stories"
  ]

  return (
    <div className="bg-white">
      {/* Hero Section */}
      <section className="pt-24 pb-12 bg-[#1E3A8A] text-white">
        <div className="max-w-7xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Blog</h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Insights, updates, and stories from the Networkli team
            </p>
          </motion.div>
        </div>
      </section>

      {/* Featured Post */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-gray-50 rounded-xl overflow-hidden shadow-lg"
          >
            <div className="md:flex">
              <div className="md:w-1/2">
                <img
                  src={featuredPost.image}
                  alt={featuredPost.title}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="md:w-1/2 p-8">
                <div className="text-sm text-connection-blue font-semibold mb-2">
                  {featuredPost.category}
                </div>
                <h2 className="text-3xl font-bold mb-4">{featuredPost.title}</h2>
                <p className="text-gray-600 mb-6">{featuredPost.excerpt}</p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="text-sm text-gray-500">
                      <span>{featuredPost.author}</span>
                      <span className="mx-2">•</span>
                      <span>{featuredPost.date}</span>
                    </div>
                  </div>
                  <Link
                    href="#"
                    className="text-connection-blue hover:text-connection-blue-70 font-semibold"
                  >
                    Read More →
                  </Link>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Categories and Recent Posts */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="md:flex md:gap-8">
            {/* Categories Sidebar */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="md:w-1/4 mb-8 md:mb-0"
            >
              <h3 className="text-xl font-bold mb-4">Categories</h3>
              <ul className="space-y-2">
                {categories.map((category, index) => (
                  <li key={index}>
                    <Link
                      href="#"
                      className="text-gray-600 hover:text-connection-blue transition-colors"
                    >
                      {category}
                    </Link>
                  </li>
                ))}
              </ul>
            </motion.div>

            {/* Recent Posts */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="md:w-3/4"
            >
              <h3 className="text-xl font-bold mb-6">Recent Posts</h3>
              <div className="space-y-8">
                {recentPosts.map((post, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="bg-white p-6 rounded-lg shadow-sm"
                  >
                    <div className="text-sm text-connection-blue font-semibold mb-2">
                      {post.category}
                    </div>
                    <h4 className="text-xl font-bold mb-2">{post.title}</h4>
                    <p className="text-gray-600 mb-4">{post.excerpt}</p>
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-gray-500">
                        <span>{post.author}</span>
                        <span className="mx-2">•</span>
                        <span>{post.date}</span>
                      </div>
                      <Link
                        href="#"
                        className="text-connection-blue hover:text-connection-blue-70 font-semibold"
                      >
                        Read More →
                      </Link>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  )
} 