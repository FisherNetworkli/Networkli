'use client'

import React from 'react'

interface PublicPageWrapperProps {
  children: React.ReactNode
}

export default function PublicPageWrapper({ children }: PublicPageWrapperProps) {
  return (
    <div className="cursor-default select-none">
      {children}
    </div>
  )
} 