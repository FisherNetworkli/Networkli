import React from 'react';
import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import LoginForm from './LoginForm';
import { authOptions } from '../api/auth/[...nextauth]/auth';
import Image from 'next/image';

export default async function LoginPage() {
  const session = await getServerSession(authOptions);

  if (session) {
    // Redirect based on user role
    if (session.user?.role === 'ADMIN') {
      redirect('/admin');
    } else {
      redirect('/dashboard');
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
      <div className="w-full max-w-md px-8 py-12 bg-white rounded-2xl shadow-sm">
        <div className="flex flex-col items-center mb-8">
          <div className="mb-6">
            <Image 
              src="/logos/networkli-logo-blue.png"
              alt="Networkli Logo" 
              width={180}
              height={48} 
              className="w-auto h-12"
              priority
            />
          </div>
          <h2 className="text-2xl font-semibold text-gray-900 tracking-tight">
            Welcome back
          </h2>
          <p className="mt-2 text-sm text-gray-500 text-center">
            Sign in to your account to continue
          </p>
        </div>
        <LoginForm />
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            Don't have an account?{' '}
            <a href="/signup" className="font-medium text-connection-blue hover:text-connection-blue-dark transition-colors duration-200">
              Sign up
            </a>
          </p>
        </div>
      </div>
    </div>
  );
} 