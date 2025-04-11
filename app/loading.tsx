"use client"

import React from 'react'
import Logo from '../components/Logo'

export default function Loading() {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-white">
      <div className="animate-pulse">
        <Logo variant="icon" className="w-16 h-16" />
      </div>
    </div>
  )
} 