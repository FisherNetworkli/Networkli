"use client"

import React from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import Image from 'next/image'
import { blogPosts } from './blogData'

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
    <div className="min-h-screen bg-white">
      {/* Hero Section */}
      <div className="bg-connection-blue text-white py-20">
        <div className="container mx-auto px-4">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">Networkli Blog</h1>
          <p className="text-xl md:text-2xl max-w-3xl">
            Insights, strategies, and stories about professional networking, career development, and building meaningful connections.
          </p>
        </div>
      </div>

      {/* Featured Post */}
      <div className="container mx-auto px-4 py-12">
        <div className="grid md:grid-cols-2 gap-8 items-center">
          <div className="relative h-[400px] rounded-lg overflow-hidden">
            <Image
              src={blogPosts[0].image}
              alt={blogPosts[0].title}
              fill
              className="object-cover"
            />
          </div>
          <div>
            <div className="flex items-center gap-4 mb-4">
              <span className="bg-connection-blue/10 text-connection-blue px-3 py-1 rounded-full text-sm">
                {blogPosts[0].category}
              </span>
              <span className="text-gray-500">{blogPosts[0].readTime}</span>
            </div>
            <h2 className="text-3xl font-bold mb-4">{blogPosts[0].title}</h2>
            <p className="text-gray-600 mb-6">{blogPosts[0].excerpt}</p>
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-full bg-gray-200"></div>
              <div>
                <p className="font-medium">{blogPosts[0].author}</p>
                <p className="text-sm text-gray-500">{blogPosts[0].date}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Blog Grid */}
      <div className="container mx-auto px-4 py-12">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {blogPosts.slice(1).map((post) => (
            <Link href={`/blog/${post.slug}`} key={post.id} className="group">
              <div className="bg-white rounded-lg overflow-hidden shadow-lg transition-transform duration-300 group-hover:-translate-y-2">
                <div className="relative h-48">
                  <Image
                    src={post.image}
                    alt={post.title}
                    fill
                    className="object-cover"
                  />
                </div>
                <div className="p-6">
                  <div className="flex items-center gap-4 mb-4">
                    <span className="bg-connection-blue/10 text-connection-blue px-3 py-1 rounded-full text-sm">
                      {post.category}
                    </span>
                    <span className="text-gray-500">{post.readTime}</span>
                  </div>
                  <h3 className="text-xl font-bold mb-3 group-hover:text-connection-blue transition-colors">
                    {post.title}
                  </h3>
                  <p className="text-gray-600 mb-4 line-clamp-2">{post.excerpt}</p>
                  <div className="flex items-center gap-4">
                    <div className="w-8 h-8 rounded-full bg-gray-200"></div>
                    <div>
                      <p className="font-medium text-sm">{post.author}</p>
                      <p className="text-xs text-gray-500">{post.date}</p>
                    </div>
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Newsletter Section */}
      <div className="bg-gray-50 py-16">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold mb-4">Stay Connected</h2>
          <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
            Subscribe to our newsletter for the latest insights on professional networking, career development, and building meaningful connections.
          </p>
          <form className="max-w-md mx-auto flex gap-4">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-connection-blue"
            />
            <button
              type="submit"
              className="bg-connection-blue text-white px-6 py-2 rounded-lg hover:bg-connection-blue/90 transition-colors"
            >
              Subscribe
            </button>
          </form>
        </div>
      </div>
    </div>
  )
} 