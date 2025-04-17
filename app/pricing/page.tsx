"use client"

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { Button } from "../components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../components/ui/card"
import { Check, Loader2, ArrowRight } from "lucide-react"
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'

export default function PricingPage() {
  const router = useRouter()
  const [userEmail, setUserEmail] = useState<string | null>(null)
  const [authStatus, setAuthStatus] = useState<'loading' | 'authenticated' | 'unauthenticated'>('loading')
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const supabase = createClientComponentClient()

  useEffect(() => {
    const getUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setUserEmail(session.user.email || null);
        setAuthStatus('authenticated');
      } else {
        setAuthStatus('unauthenticated');
      }
    };
    getUser();
  }, [supabase]);

  const handleUpgrade = async (plan: string) => {
    if (authStatus !== 'authenticated') {
      router.push('/login?redirect=/pricing');
      return;
    }

    try {
      setIsLoading(true)
      setSelectedPlan(plan)
      
      const response = await fetch('/api/create-checkout-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          plan,
          email: userEmail
        }),
      })
      
      if (!response.ok) {
        throw new Error('Failed to create checkout session')
      }
      
      const data = await response.json()
      
      if (data.isContactForm) {
        router.push(data.url)
      } else if (data.url) {
        window.location.href = data.url
      }
    } catch (error) {
      console.error('Error:', error)
      // You might want to show an error toast here
    } finally {
      setIsLoading(false)
      setSelectedPlan(null)
    }
  }

  return (
    <div className="bg-white">
      {/* Hero Section */}
      <section className="pt-32 pb-16 bg-gradient-to-b from-white to-gray-50">
        <div className="max-w-5xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-6xl md:text-7xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600">
              Choose Your Plan
            </h1>
            <p className="text-xl text-gray-500 max-w-2xl mx-auto">
              Simple, transparent pricing for everyone. No hidden fees.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Pricing Tiers */}
      <section className="py-16">
        <div className="max-w-5xl mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-8">
            {/* Free Plan */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <Card className="h-full border-0 shadow-sm hover:shadow-md transition-shadow duration-300">
                <CardHeader>
                  <CardTitle className="text-2xl font-medium">Free</CardTitle>
                  <CardDescription className="text-base">Basic features for getting started</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold mb-6">$0<span className="text-lg font-normal text-gray-500">/month</span></div>
                  <ul className="space-y-4">
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Basic profile
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Limited connections
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Basic search
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Community access
                    </li>
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button 
                    className="w-full bg-gray-100 text-gray-900 hover:bg-gray-200 transition-colors duration-300"
                    onClick={() => router.push('/signup')}
                  >
                    Get Started
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>

            {/* Premium Plan */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <Card className={`h-full border-2 ${selectedPlan === 'premium' ? 'border-blue-500' : 'border-gray-200'} shadow-md hover:shadow-lg transition-all duration-300 relative`}>
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <span className="bg-blue-500 text-white px-4 py-1 rounded-full text-sm font-medium">Most Popular</span>
                </div>
                <CardHeader>
                  <CardTitle className="text-2xl font-medium">Premium</CardTitle>
                  <CardDescription className="text-base">Everything you need to succeed</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold mb-6">$12.99<span className="text-lg font-normal text-gray-500">/month</span></div>
                  <ul className="space-y-4">
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Advanced profile features
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Unlimited connections
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Priority search results
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Analytics dashboard
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Premium support
                    </li>
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button 
                    className={`w-full ${selectedPlan === 'premium' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white transition-colors duration-300`}
                    onClick={() => handleUpgrade('premium')}
                    disabled={isLoading}
                  >
                    {isLoading && selectedPlan === 'premium' ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      'Upgrade to Premium'
                    )}
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>

            {/* Organizer Plan */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              viewport={{ once: true }}
            >
              <Card className={`h-full border-2 ${selectedPlan === 'organizer' ? 'border-purple-500' : 'border-gray-200'} shadow-md hover:shadow-lg transition-all duration-300`}>
                <CardHeader>
                  <CardTitle className="text-2xl font-medium">Organizer</CardTitle>
                  <CardDescription className="text-base">For creating groups and events</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold mb-6">$99<span className="text-lg font-normal text-gray-500">/month</span></div>
                  <ul className="space-y-4">
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      All Premium features
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Create unlimited groups
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Host events and meetups
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Advanced analytics
                    </li>
                    <li className="flex items-center text-gray-600">
                      <Check className="mr-3 h-5 w-5 text-green-500" />
                      Priority support
                    </li>
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button 
                    className={`w-full ${selectedPlan === 'organizer' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700'} text-white transition-colors duration-300`}
                    onClick={() => handleUpgrade('organizer')}
                    disabled={isLoading}
                  >
                    {isLoading && selectedPlan === 'organizer' ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      'Upgrade to Organizer'
                    )}
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Enterprise Section */}
      <section className="py-24 bg-gradient-to-b from-gray-50 to-white">
        <div className="max-w-5xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6">Enterprise Solutions</h2>
            <p className="text-xl text-gray-500 max-w-3xl mx-auto">
              Custom solutions for organizations that need more than standard plans
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-12">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
            >
              <h3 className="text-2xl font-semibold mb-4">Enterprise Sales</h3>
              <p className="text-gray-600 mb-6 leading-relaxed">
                Custom enterprise solutions with dedicated support, advanced security features, and tailored implementation for your organization's specific needs.
              </p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-start">
                  <Check className="mr-3 h-5 w-5 text-green-500 mt-0.5" />
                  <span className="text-gray-600">Dedicated account management</span>
                </li>
                <li className="flex items-start">
                  <Check className="mr-3 h-5 w-5 text-green-500 mt-0.5" />
                  <span className="text-gray-600">Custom integration options</span>
                </li>
                <li className="flex items-start">
                  <Check className="mr-3 h-5 w-5 text-green-500 mt-0.5" />
                  <span className="text-gray-600">Advanced security and compliance</span>
                </li>
                <li className="flex items-start">
                  <Check className="mr-3 h-5 w-5 text-green-500 mt-0.5" />
                  <span className="text-gray-600">Priority support and training</span>
                </li>
              </ul>
              <Button 
                className="w-full bg-gray-900 text-white hover:bg-gray-800 transition-colors duration-300"
                onClick={() => router.push('/contact?type=enterprise')}
              >
                Contact Enterprise Sales
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
              className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
            >
              <h3 className="text-2xl font-semibold mb-4">Algorithm Licensing</h3>
              <p className="text-gray-600 mb-6 leading-relaxed">
                License our proprietary AI matching algorithm to power your own applications and services. Perfect for companies looking to integrate our technology into their existing platforms.
              </p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-start">
                  <Check className="mr-3 h-5 w-5 text-green-500 mt-0.5" />
                  <span className="text-gray-600">API access to our matching algorithm</span>
                </li>
                <li className="flex items-start">
                  <Check className="mr-3 h-5 w-5 text-green-500 mt-0.5" />
                  <span className="text-gray-600">Custom implementation support</span>
                </li>
                <li className="flex items-start">
                  <Check className="mr-3 h-5 w-5 text-green-500 mt-0.5" />
                  <span className="text-gray-600">Scalable pricing based on usage</span>
                </li>
                <li className="flex items-start">
                  <Check className="mr-3 h-5 w-5 text-green-500 mt-0.5" />
                  <span className="text-gray-600">Technical documentation and support</span>
                </li>
              </ul>
              <Button 
                className="w-full bg-gray-900 text-white hover:bg-gray-800 transition-colors duration-300"
                onClick={() => router.push('/contact?type=licensing')}
              >
                Inquire About Licensing
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 bg-gradient-to-b from-gray-50 to-white">
        <div className="max-w-3xl mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-16">Frequently Asked Questions</h2>
          <div className="space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
            >
              <h3 className="text-xl font-semibold mb-3">Can I change plans later?</h3>
              <p className="text-gray-600 leading-relaxed">
                Yes, you can upgrade or downgrade your plan at any time. Changes will be reflected in your next billing cycle.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
            >
              <h3 className="text-xl font-semibold mb-3">What payment methods do you accept?</h3>
              <p className="text-gray-600 leading-relaxed">
                We accept all major credit cards through our secure Stripe payment processing.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
            >
              <h3 className="text-xl font-semibold mb-3">Is there a free trial?</h3>
              <p className="text-gray-600 leading-relaxed">
                Yes, you can start with our Free plan to explore basic features. Upgrade anytime to access premium features.
              </p>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  )
} 