"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'
import PublicPageWrapper from '../components/PublicPageWrapper'

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [status, setStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
  const [error, setError] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setStatus('submitting');
    setError('');

    try {
      const response = await fetch('/api/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to submit form');
      }

      setStatus('success');
      setFormData({
        name: '',
        email: '',
        subject: '',
        message: ''
      });
    } catch (error) {
      setError('Failed to send message. Please try again.');
    }
  };

  return (
    <PublicPageWrapper>
      <div className="min-h-screen bg-white">
        {/* Hero Section */}
        <section className="relative h-[40vh] min-h-[300px]">
          <Image
            src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images//20250412_1339_Joyful%20Message%20Sent_simple_compose_01jrnr3jwsfa6sefbmxw855dsz.png"
            alt="Contact Us"
            fill
            className="object-cover"
            sizes="(max-width: 768px) 100vw, 50vw"
          />
          <div className="absolute inset-0 bg-black/30" />
          <div className="absolute inset-0 flex items-center">
            <div className="container mx-auto px-4">
              <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
                Get in Touch
              </h1>
              <p className="text-xl text-white/90 max-w-2xl">
                We'd love to hear from you. Send us a message and we'll respond as soon as possible.
              </p>
            </div>
          </div>
        </section>

        {/* Contact Form Section */}
        <section className="py-16">
          <div className="max-w-7xl mx-auto px-4">
            <div className="grid md:grid-cols-2 gap-12 items-start">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="bg-white p-8 rounded-lg shadow-sm border border-gray-200"
              >
                <h2 className="text-3xl font-bold mb-6">Send us a message</h2>
                {status === 'success' && (
                  <div className="mb-6 p-4 bg-green-50 text-green-800 rounded-lg">
                    Thank you for your message! We'll get back to you soon.
                  </div>
                )}
                {status === 'error' && (
                  <div className="mb-6 p-4 bg-red-50 text-red-800 rounded-lg">
                    {error}
                  </div>
                )}
                <form className="space-y-6" onSubmit={handleSubmit}>
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                      Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-connection-blue focus:border-transparent"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                      Email
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-connection-blue focus:border-transparent"
                      placeholder="your@email.com"
                    />
                  </div>
                  <div>
                    <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-1">
                      Subject
                    </label>
                    <input
                      type="text"
                      id="subject"
                      name="subject"
                      value={formData.subject}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-connection-blue focus:border-transparent"
                      placeholder="How can we help?"
                    />
                  </div>
                  <div>
                    <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-1">
                      Message
                    </label>
                    <textarea
                      id="message"
                      name="message"
                      value={formData.message}
                      onChange={handleChange}
                      required
                      rows={4}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-connection-blue focus:border-transparent"
                      placeholder="Your message..."
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={status === 'submitting'}
                    className="w-full bg-connection-blue text-white py-3 rounded-full hover:bg-connection-blue-70 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {status === 'submitting' ? 'Sending...' : 'Send Message'}
                  </button>
                </form>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="space-y-8"
              >
                <div>
                  <h3 className="text-2xl font-bold mb-4">Contact Information</h3>
                  <p className="text-gray-600 mb-4">
                    Have questions about Networkli? We're here to help you connect meaningfully.
                  </p>
                  <div className="space-y-4">
                    <div className="flex items-start">
                      <span className="text-networkli-orange mr-2">•</span>
                      <div>
                        <p className="font-medium">Email</p>
                        <p className="text-gray-600">support@networkli.com</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <span className="text-networkli-orange mr-2">•</span>
                      <div>
                        <p className="font-medium">Support Hours</p>
                        <p className="text-gray-600">Monday - Friday, 9am - 5pm EST</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-2xl font-bold mb-4">Enterprise Support</h3>
                  <p className="text-gray-600 mb-4">
                    Looking for enterprise solutions or API integration? Our sales team is ready to help.
                  </p>
                  <button className="bg-black text-white px-6 py-3 rounded-full hover:bg-gray-800 transition-colors">
                    Contact Sales
                  </button>
                </div>

                <div>
                  <h3 className="text-2xl font-bold mb-4">Quick Links</h3>
                  <div className="space-y-2">
                    <p className="text-connection-blue hover:underline cursor-pointer">
                      Documentation
                    </p>
                    <p className="text-connection-blue hover:underline cursor-pointer">
                      API Reference
                    </p>
                    <p className="text-connection-blue hover:underline cursor-pointer">
                      FAQs
                    </p>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Support Options */}
        <section className="py-16 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">How Can We Help?</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm"
              >
                <h3 className="text-xl font-bold mb-4">General Support</h3>
                <p className="text-gray-600">
                  Questions about your account, billing, or general platform usage
                </p>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm"
              >
                <h3 className="text-xl font-bold mb-4">Technical Support</h3>
                <p className="text-gray-600">
                  API integration, implementation assistance, and technical documentation
                </p>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-lg shadow-sm"
              >
                <h3 className="text-xl font-bold mb-4">Enterprise Solutions</h3>
                <p className="text-gray-600">
                  Custom implementations, large-scale deployments, and business partnerships
                </p>
              </motion.div>
            </div>
          </div>
        </section>
      </div>
    </PublicPageWrapper>
  )
} 