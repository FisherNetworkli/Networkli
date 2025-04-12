"use client"

import React from 'react'
import { motion } from 'framer-motion'

export default function PricingPage() {
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
            <h1 className="text-5xl md:text-6xl font-bold mb-6">Simple, Transparent Pricing</h1>
            <p className="text-xl text-gray-100 max-w-3xl mx-auto">
              Choose the plan that best fits your networking needs
            </p>
          </motion.div>
        </div>
      </section>

      {/* Pricing Tiers */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-8">
            {/* Basic Plan */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="bg-white p-8 rounded-lg shadow-sm border border-gray-200"
            >
              <h3 className="text-2xl font-bold mb-4">Basic</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold">$9</span>
                <span className="text-gray-600">/month</span>
              </div>
              <p className="text-gray-600 mb-6">Perfect for individuals starting their networking journey</p>
              <ul className="space-y-4 mb-8">
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>AI-powered matching</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Basic conversation starters</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Up to 20 connections/month</span>
                </li>
              </ul>
              <button className="w-full bg-black text-white py-3 rounded-full hover:bg-gray-800 transition-colors">
                Get Started
              </button>
            </motion.div>

            {/* Professional Plan */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
              className="bg-connection-blue text-white p-8 rounded-lg shadow-lg transform scale-105"
            >
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                <span className="bg-networkli-orange text-white px-4 py-1 rounded-full text-sm">Most Popular</span>
              </div>
              <h3 className="text-2xl font-bold mb-4">Professional</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold">$29</span>
                <span>/month</span>
              </div>
              <p className="text-gray-100 mb-6">Ideal for active networkers and professionals</p>
              <ul className="space-y-4 mb-8">
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Everything in Basic</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Advanced AI matching</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Unlimited connections</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Analytics dashboard</span>
                </li>
              </ul>
              <button className="w-full bg-white text-connection-blue py-3 rounded-full hover:bg-gray-100 transition-colors">
                Get Started
              </button>
            </motion.div>

            {/* Enterprise Plan */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              viewport={{ once: true }}
              className="bg-white p-8 rounded-lg shadow-sm border border-gray-200"
            >
              <h3 className="text-2xl font-bold mb-4">Enterprise</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold">Custom</span>
              </div>
              <p className="text-gray-600 mb-6">For organizations and large events</p>
              <ul className="space-y-4 mb-8">
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Everything in Professional</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>API access</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Custom implementation</span>
                </li>
                <li className="flex items-start">
                  <span className="text-networkli-orange mr-2">✓</span>
                  <span>Dedicated support</span>
                </li>
              </ul>
              <button className="w-full bg-black text-white py-3 rounded-full hover:bg-gray-800 transition-colors">
                Contact Sales
              </button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Annual Pricing */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl font-bold mb-6">Save with Annual Plans</h2>
            <p className="text-xl text-gray-600 mb-8">
              Get two months free when you choose annual billing
            </p>
            <div className="inline-flex items-center bg-white rounded-full p-1 border border-gray-200">
              <button className="px-6 py-2 rounded-full bg-connection-blue text-white">
                Monthly
              </button>
              <button className="px-6 py-2 rounded-full text-gray-600">
                Annual
              </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Enterprise Features */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-6">Enterprise Features</h2>
            <p className="text-xl text-gray-600">
              Customizable solutions for organizations of any size
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="p-6 bg-white rounded-lg shadow-sm border border-gray-100"
            >
              <h3 className="text-xl font-bold mb-4">API Integration</h3>
              <p className="text-gray-600">
                Integrate our AI matching algorithm into your existing platforms
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
              className="p-6 bg-white rounded-lg shadow-sm border border-gray-100"
            >
              <h3 className="text-xl font-bold mb-4">Event Solutions</h3>
              <p className="text-gray-600">
                Perfect for conferences and large networking events
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              viewport={{ once: true }}
              className="p-6 bg-white rounded-lg shadow-sm border border-gray-100"
            >
              <h3 className="text-xl font-bold mb-4">Custom Support</h3>
              <p className="text-gray-600">
                Dedicated account management and priority support
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-3xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">Frequently Asked Questions</h2>
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="bg-white p-6 rounded-lg shadow-sm"
            >
              <h3 className="text-xl font-bold mb-2">Can I change plans later?</h3>
              <p className="text-gray-600">
                Yes, you can upgrade or downgrade your plan at any time. Changes will be reflected in your next billing cycle.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="bg-white p-6 rounded-lg shadow-sm"
            >
              <h3 className="text-xl font-bold mb-2">What's included in the API access?</h3>
              <p className="text-gray-600">
                Enterprise plans include full access to our AI matching algorithm API, documentation, and implementation support.
              </p>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  )
} 