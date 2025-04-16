import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs'
import { cookies } from 'next/headers'
import { NextResponse } from 'next/server'
import { supabase, supabaseAdmin } from '@/utils/supabase'

export async function GET() {
  try {
    // Try to get auth settings using admin client
    const { data: settingsData, error: settingsError } = await supabaseAdmin?.auth.admin.listUsers() || { data: null, error: new Error('Admin client not initialized') }
    
    // Try to get schema info
    const { data: schemaData, error: schemaError } = await supabaseAdmin?.from('auth.users').select('*') || { data: null, error: new Error('Admin client not initialized') }

    return NextResponse.json({
      status: 'checking',
      settings: {
        data: settingsData,
        error: settingsError ? { message: settingsError.message } : null
      },
      schema: {
        data: schemaData,
        error: schemaError ? { message: schemaError.message } : null
      }
    })
  } catch (error) {
    console.error('Auth test error:', error)
    return NextResponse.json({
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
} 