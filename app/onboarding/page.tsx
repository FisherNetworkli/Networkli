import { Metadata } from 'next';
import { redirect } from 'next/navigation';
import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import OnboardingForm from '@/app/components/auth/OnboardingForm';

export const metadata: Metadata = {
  title: 'Complete Your Profile | Networkli',
  description: 'Complete your profile to get the most out of Networkli',
};

export default async function OnboardingPage() {
  const supabase = createServerComponentClient({ cookies });
  
  // Check if user is authenticated
  const { data: { session } } = await supabase.auth.getSession();
  
  if (!session) {
    redirect('/login');
  }
  
  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-extrabold text-gray-900">Complete Your Profile</h1>
          <p className="mt-2 text-lg text-gray-600">
            Tell us more about yourself to get personalized recommendations
          </p>
        </div>
        
        <OnboardingForm user={session.user} />
      </div>
    </div>
  );
} 