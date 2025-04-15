import { NextResponse } from 'next/server'
import { getToken } from 'next-auth/jwt'
import { NextRequestWithAuth } from 'next-auth/middleware'

// Define public paths that don't require authentication
const publicPaths = [
  '/',
  '/about',
  '/features',
  '/pricing',
  '/blog',
  '/signin',
  '/signup',
  '/api/auth',
  '/_next',
  '/favicon.ico',
  '/images',
  '/fonts',
  '/styles'
]

// Define auth paths that should redirect to dashboard if already authenticated
const authPaths = ['/signin', '/signup']

// Define protected paths that require authentication
const protectedPaths = ['/dashboard', '/profile', '/settings']

export default async function middleware(req: NextRequestWithAuth) {
  const token = await getToken({ req })
  const { pathname } = req.nextUrl

  // Check if the path is a public path (no auth required)
  const isPublicPath = publicPaths.some(path => pathname.startsWith(path))
  
  // Check if the path is an auth page (signin/signup)
  const isAuthPage = authPaths.some(path => pathname.startsWith(path))

  // Check if the path is a protected route
  const isProtectedPath = protectedPaths.some(path => pathname.startsWith(path))

  // If user is authenticated and tries to access auth pages, redirect to dashboard
  if (token && isAuthPage) {
    return NextResponse.redirect(new URL('/dashboard', req.url))
  }

  // If user is not authenticated and tries to access protected routes, redirect to signin
  if (!token && isProtectedPath) {
    const signInUrl = new URL('/signin', req.url)
    signInUrl.searchParams.set('callbackUrl', pathname)
    return NextResponse.redirect(signInUrl)
  }

  // Allow access to all other routes
  return NextResponse.next()
}

// Configure which paths the middleware should run on
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api/auth (auth API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api/auth|_next/static|_next/image|favicon.ico).*)'
  ]
} 