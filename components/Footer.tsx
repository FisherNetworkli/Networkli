"use client"

import React from 'react'
import Link from 'next/link'
import Logo from './Logo'

export default function Footer() {
  return (
    <footer className="bg-gray-50 border-t border-gray-200 relative z-10">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="col-span-1 md:col-span-1">
            <Link href="/" className="inline-block">
              <Logo variant="black" className="w-32 h-8" />
            </Link>
            <p className="mt-4 text-sm text-gray-500">
              Professional networking reimagined for the modern world.
            </p>
          </div>
          
          <nav className="relative">
            <h3 className="text-sm font-semibold text-gray-900 tracking-wider uppercase">Product</h3>
            <ul className="mt-4 space-y-4">
              <li className="block">
                <Link 
                  href="/features" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Features
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/pricing" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Pricing
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/download" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Download
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/security" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Security
                </Link>
              </li>
            </ul>
          </nav>
          
          <nav className="relative">
            <h3 className="text-sm font-semibold text-gray-900 tracking-wider uppercase">Company</h3>
            <ul className="mt-4 space-y-4">
              <li className="block">
                <Link 
                  href="/about" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  About
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/blog" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Blog
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/careers" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Careers
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/contact" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Contact
                </Link>
              </li>
            </ul>
          </nav>
          
          <nav className="relative">
            <h3 className="text-sm font-semibold text-gray-900 tracking-wider uppercase">Legal</h3>
            <ul className="mt-4 space-y-4">
              <li className="block">
                <Link 
                  href="/privacy" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Privacy Policy
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/terms" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Terms of Service
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/cookies" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Cookie Policy
                </Link>
              </li>
              <li className="block">
                <Link 
                  href="/accessibility" 
                  className="inline-block text-base text-gray-500 hover:text-connection-blue transition-colors cursor-pointer relative z-10"
                >
                  Accessibility
                </Link>
              </li>
            </ul>
          </nav>
        </div>
        
        <div className="mt-12 border-t border-gray-200 pt-8">
          <p className="text-base text-gray-400 text-center">
            &copy; {new Date().getFullYear()} Networkli. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
} 