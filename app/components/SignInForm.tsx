'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { AuthError } from '@supabase/supabase-js';
import { supabase } from '@/utils/supabase';

export default function SignInForm() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      // First, check if Supabase is properly initialized
      if (!supabase.auth) {
        throw new Error('Supabase client not properly initialized');
      }

      // Log the attempt
      console.log('Attempting to sign in...', {
        email: formData.email,
        url: process.env.NEXT_PUBLIC_SUPABASE_URL
      });

      // Try to get the current session first
      const { data: sessionData } = await supabase.auth.getSession();
      console.log('Current session:', sessionData);

      // Attempt sign in
      const { data, error: signInError } = await supabase.auth.signInWithPassword({
        email: formData.email,
        password: formData.password,
      });

      // Log the complete response for debugging
      console.log('Sign in response:', {
        data,
        error: signInError ? {
          message: signInError.message,
          status: signInError.status,
          name: signInError.name
        } : null
      });

      if (signInError) {
        throw signInError;
      }

      if (!data?.user) {
        throw new Error('No user data returned');
      }

      // Clear form and redirect
      setFormData({ email: '', password: '' });
      router.refresh();
      router.push('/dashboard');
    } catch (err) {
      console.error('Detailed error:', err);
      
      if (err instanceof AuthError) {
        switch (err.message) {
          case 'Invalid login credentials':
            setError('Invalid email or password');
            break;
          case 'Database error querying schema':
            setError('Authentication system is temporarily unavailable. Please try again in a few minutes. If the problem persists, contact support.');
            // Log additional details for debugging
            console.error('Schema error details:', {
              status: err.status,
              name: err.name,
              stack: err.stack
            });
            break;
          default:
            setError(`Authentication error: ${err.message}`);
        }
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="bg-red-50 text-red-500 p-3 rounded-md text-sm">
          {error}
        </div>
      )}

      <div>
        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
          Email
        </label>
        <input
          type="email"
          id="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          required
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
        />
      </div>

      <div>
        <label htmlFor="password" className="block text-sm font-medium text-gray-700">
          Password
        </label>
        <input
          type="password"
          id="password"
          name="password"
          value={formData.password}
          onChange={handleChange}
          required
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="text-sm">
          <Link href="/forgot-password" className="font-medium text-indigo-600 hover:text-indigo-500">
            Forgot your password?
          </Link>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
      >
        {loading ? 'Signing in...' : 'Sign In'}
      </button>

      <div className="text-center text-sm">
        <span className="text-gray-600">Don't have an account? </span>
        <Link href="/signup" className="font-medium text-indigo-600 hover:text-indigo-500">
          Sign up
        </Link>
      </div>
    </form>
  );
} 