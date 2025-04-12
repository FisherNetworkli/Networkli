"use client"

import React from 'react'
import Logo from '../components/Logo'

export default function Loading() {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-white">
      <div className="flex flex-col items-center space-y-4">
        <Logo 
          variant="default"
          className="w-32 h-8 animate-pulse"
        />
        <div className="text-sm text-gray-500">Loading...</div>
      </div>
    </div>
  )
} 