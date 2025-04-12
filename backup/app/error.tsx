'use client'

import React from 'react'
import Logo from '../components/Logo'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-white px-4">
      <div className="max-w-md w-full space-y-8 text-center">
        <div className="flex justify-center">
          <Logo 
            variant="default"
            className="w-32 h-8"
          />
        </div>
        <h2 className="mt-6 text-3xl font-bold text-gray-900">
          Something went wrong!
        </h2>
        <p className="mt-2 text-sm text-gray-600">
          {error.message || 'An unexpected error occurred. Please try again.'}
        </p>
        <div className="mt-6">
          <button
            onClick={reset}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-connection-blue hover:bg-connection-blue-70 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue"
          >
            Try again
          </button>
        </div>
      </div>
    </div>
  )
} 