"use client"

import React, { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import Logo from './Logo'

export default function Navbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const handleLinkClick = () => {
    setIsMobileMenuOpen(false)
  }

  return (
    <header 
      className="fixed top-0 left-0 right-0 z-50 bg-white shadow-sm"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <Logo 
              variant="default"
              className="w-48 h-8"
            />
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            <Link 
              href="/features" 
              className="text-sm font-medium text-gray-600 hover:text-connection-blue transition-colors"
            >
              Features
            </Link>
            <Link 
              href="/pricing" 
              className="text-sm font-medium text-gray-600 hover:text-connection-blue transition-colors"
            >
              Pricing
            </Link>
            <Link 
              href="/blog" 
              className="text-sm font-medium text-gray-600 hover:text-connection-blue transition-colors"
            >
              Blog
            </Link>
            <Link 
              href="/about" 
              className="text-sm font-medium text-gray-600 hover:text-connection-blue transition-colors"
            >
              About
            </Link>
            <Link 
              href="/contact" 
              className="text-sm font-medium text-gray-600 hover:text-connection-blue transition-colors"
            >
              Contact
            </Link>
          </nav>

          {/* CTA Buttons */}
          <div className="hidden md:flex items-center space-x-6">
            <Link 
              href="/login" 
              className="text-sm font-medium text-gray-600 hover:text-connection-blue transition-colors"
            >
              Log in
            </Link>
            <Link 
              href="/signup" 
              className="inline-flex items-center justify-center px-6 py-2 rounded-full text-sm font-medium bg-connection-blue text-white hover:bg-connection-blue-70 transition-colors"
            >
              Sign up
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              type="button"
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-600 hover:text-connection-blue hover:bg-gray-100 transition-colors"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              <span className="sr-only">Open main menu</span>
              {isMobileMenuOpen ? (
                <XMarkIcon className="h-6 w-6" aria-hidden="true" />
              ) : (
                <Bars3Icon className="h-6 w-6" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isMobileMenuOpen && (
        <motion.div 
          className="md:hidden bg-white shadow-lg"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
        >
          <div className="px-4 pt-4 pb-6 space-y-4">
            <div className="flex items-center justify-center mb-6">
              <Logo 
                variant="default"
                className="w-32 h-8"
              />
            </div>
            <Link 
              href="/features" 
              onClick={handleLinkClick}
              className="block px-4 py-2.5 text-base font-medium text-gray-600 hover:text-connection-blue hover:bg-gray-50 rounded-lg transition-colors"
            >
              Features
            </Link>
            <Link 
              href="/pricing" 
              onClick={handleLinkClick}
              className="block px-4 py-2.5 text-base font-medium text-gray-600 hover:text-connection-blue hover:bg-gray-50 rounded-lg transition-colors"
            >
              Pricing
            </Link>
            <Link 
              href="/blog" 
              onClick={handleLinkClick}
              className="block px-4 py-2.5 text-base font-medium text-gray-600 hover:text-connection-blue hover:bg-gray-50 rounded-lg transition-colors"
            >
              Blog
            </Link>
            <Link 
              href="/about" 
              onClick={handleLinkClick}
              className="block px-4 py-2.5 text-base font-medium text-gray-600 hover:text-connection-blue hover:bg-gray-50 rounded-lg transition-colors"
            >
              About
            </Link>
            <Link 
              href="/contact" 
              onClick={handleLinkClick}
              className="block px-4 py-2.5 text-base font-medium text-gray-600 hover:text-connection-blue hover:bg-gray-50 rounded-lg transition-colors"
            >
              Contact
            </Link>
            <div className="pt-4 space-y-2">
              <Link 
                href="/login" 
                onClick={handleLinkClick}
                className="block w-full text-center px-4 py-2.5 text-base font-medium text-gray-600 hover:text-connection-blue hover:bg-gray-50 rounded-lg transition-colors"
              >
                Log in
              </Link>
              <Link 
                href="/signup" 
                onClick={handleLinkClick}
                className="block w-full text-center px-4 py-2.5 text-base font-medium bg-connection-blue text-white hover:bg-connection-blue-70 rounded-lg transition-colors"
              >
                Sign up
              </Link>
            </div>
          </div>
        </motion.div>
      )}
    </header>
  )
} 