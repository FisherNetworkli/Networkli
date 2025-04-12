"use client"

import React from 'react'
import Link from 'next/link'
import Logo from './Logo'

export default function AppBanner() {
  return (
    <div className="bg-connection-blue text-white">
      <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Logo variant="icon" className="w-8 h-8 mr-3" />
            <p className="font-raleway text-sm">
              Get the Networkli mobile app
            </p>
          </div>
          <Link
            href="/download"
            className="inline-flex items-center px-3 py-1 border border-white rounded-full text-sm font-raleway hover:bg-white/10 transition-colors"
          >
            Download
          </Link>
        </div>
      </div>
    </div>
  )
} 