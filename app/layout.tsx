import React from 'react'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import '../styles/globals.css'
import Navbar from '../components/Navbar'
import Footer from '../components/Footer'
import Providers from '../components/Providers'
import DemoModeIndicator from '../components/DemoModeIndicator'
import { createServerComponentClient } from '@supabase/auth-helpers-nextjs'
import { cookies } from 'next/headers'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: 'Networkli - Professional Networking Reimagined for Introverts',
  description: 'Connect with purpose, build meaningful professional relationships, and grow your career with Networkli. Our AI-powered platform helps introverts network comfortably and authentically.',
  keywords: 'professional networking, introvert networking, AI matching, meaningful connections, career growth, networking app, introvert-friendly networking',
  authors: [{ name: 'Dan Fisher', url: 'https://networkli.com/about' }, { name: 'Brittany Furnari Fisher', url: 'https://networkli.com/about' }],
  creator: 'Networkli',
  publisher: 'Networkli',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://networkli.com'),
  alternates: {
    canonical: '/',
  },
  icons: {
    icon: [
      { url: '/logos/Applogo.png', sizes: '32x32' },
      { url: '/logos/Applogo.png', sizes: '16x16' },
    ],
    apple: [
      { url: '/logos/Applogo.png', sizes: '180x180' },
    ],
  },
  manifest: '/manifest.json',
  openGraph: {
    title: 'Networkli - Professional Networking Reimagined for Introverts',
    description: 'Connect with purpose, build meaningful professional relationships, and grow your career with Networkli. Our AI-powered platform helps introverts network comfortably and authentically.',
    url: 'https://networkli.com',
    siteName: 'Networkli',
    images: [
      {
        url: '/logos/networkli-logo-blue.png',
        width: 1200,
        height: 630,
        alt: 'Networkli - Professional Networking Reimagined',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Networkli - Professional Networking Reimagined for Introverts',
    description: 'Connect with purpose, build meaningful professional relationships, and grow your career with Networkli.',
    images: ['/logos/networkli-logo-blue.png'],
    creator: '@networkli',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'google-site-verification-code',
  },
}

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const supabase = createServerComponentClient({ cookies })
  const { data: { session } } = await supabase.auth.getSession()

  return (
    <html lang="en" className={`${inter.variable} scroll-smooth antialiased`}>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "SoftwareApplication",
              "name": "Networkli",
              "applicationCategory": "BusinessApplication",
              "operatingSystem": "Web, iOS, Android",
              "offers": {
                "@type": "Offer",
                "price": "0",
                "priceCurrency": "USD"
              },
              "description": "Professional networking reimagined for introverts and thoughtful professionals. Build meaningful connections at your own pace.",
              "aggregateRating": {
                "@type": "AggregateRating",
                "ratingValue": "4.8",
                "ratingCount": "120"
              }
            })
          }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "Organization",
              "name": "Networkli",
              "url": "https://networkli.com",
              "logo": "https://networkli.com/logos/networkli-logo-blue.png",
              "sameAs": [
                "https://twitter.com/networkli",
                "https://www.linkedin.com/company/networkli",
                "https://www.facebook.com/networkli"
              ],
              "contactPoint": {
                "@type": "ContactPoint",
                "telephone": "+1-555-123-4567",
                "contactType": "customer service",
                "email": "support@networkli.com"
              }
            })
          }}
        />
      </head>
      <body className="min-h-screen bg-gradient-to-b from-white via-white to-connection-blue-40/10 font-sans text-gray-900">
        <Providers>
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-networkli-orange-40/10 to-connection-blue-40/10 opacity-50" />
            <Navbar />
            <main className="relative flex min-h-screen flex-col">
              {children}
            </main>
            <Footer />
            <DemoModeIndicator />
          </div>
        </Providers>
      </body>
    </html>
  )
} 