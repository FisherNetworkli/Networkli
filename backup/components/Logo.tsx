"use client"

import React from 'react'
import Image from 'next/image'

type LogoVariant = 'default' | 'black' | 'white' | 'icon'

type LogoProps = {
  variant?: LogoVariant
  className?: string
}

export default function Logo({ variant = 'default', className = '' }: LogoProps) {
  const getLogoSrc = () => {
    switch (variant) {
      case 'black':
        return '/logos/logo black.png'
      case 'white':
        return '/logos/Logo white.png'
      case 'icon':
        return '/logos/Applogo.png'
      default:
        return '/logos/networkli-logo-blue.png'
    }
  }

  return (
    <div className={`logo-container ${variant === 'icon' ? 'icon' : ''} ${className}`}>
      <Image
        src={getLogoSrc()}
        alt="Networkli"
        fill
        className="logo-image"
        priority={true}
      />
    </div>
  )
} 