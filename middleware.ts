import { createServerClient } from '@supabase/ssr'
import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'
import { getToken } from 'next-auth/jwt'

export async function middleware(request: NextRequest) {
  let response = NextResponse.next({
    request: {
      headers: request.headers,
    },
  })

  // Check for NextAuth session
  const token = await getToken({ req: request })
  
  // If we have a NextAuth session, use that
  if (token) {
    // Add the user role to the request headers
    const requestHeaders = new Headers(request.headers)
    requestHeaders.set('x-user-role', token.role as string)
    
    response = NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    })
    
    return response
  }

  // Otherwise, try Supabase
  try {
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return request.cookies.get(name)?.value
          },
          set(name: string, value: string, options: any) {
            request.cookies.set({
              name,
              value,
              ...options,
            })
            response = NextResponse.next({
              request: {
                headers: request.headers,
              },
            })
            response.cookies.set({
              name,
              value,
              ...options,
            })
          },
          remove(name: string, options: any) {
            request.cookies.set({
              name,
              value: '',
              ...options,
            })
            response = NextResponse.next({
              request: {
                headers: request.headers,
              },
            })
            response.cookies.set({
              name,
              value: '',
              ...options,
            })
          },
        },
      }
    )

    const { data: { session } } = await supabase.auth.getSession()
    
    if (session) {
      // Add the user role to the request headers
      const requestHeaders = new Headers(request.headers)
      requestHeaders.set('x-user-role', session.user.user_metadata.role || 'USER')
      
      response = NextResponse.next({
        request: {
          headers: requestHeaders,
        },
      })
    }
  } catch (error) {
    console.error('Supabase middleware error:', error)
  }

  const isAdminRoute = request.nextUrl.pathname.startsWith('/admin')
  const isApiAdminRoute = request.nextUrl.pathname.startsWith('/api/admin')

  if ((isAdminRoute || isApiAdminRoute) && !token) {
    const loginUrl = new URL('/login', request.url)
    loginUrl.searchParams.set('callbackUrl', request.url)
    return NextResponse.redirect(loginUrl)
  }

  return response
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|public).*)',
    '/admin/:path*',
    '/api/admin/:path*',
  ],
} 