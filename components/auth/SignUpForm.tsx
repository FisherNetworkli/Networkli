'use client';

import { useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { useRouter } from 'next/navigation';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { BasicInfoStep } from './steps/BasicInfoStep';
import { PreferencesStep } from './steps/PreferencesStep';
import { motion } from 'framer-motion';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { toast } from 'sonner';

interface FormData {
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  confirmPassword: string;
  skills: string[];
  interests: string[];
  professionalGoals: string[];
  values: string[];
}

export default function SignUpForm() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const supabase = createClientComponentClient();
  
  const methods = useForm<FormData>({
    defaultValues: {
      skills: [],
      interests: [],
      professionalGoals: [],
      values: [],
    }
  });

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    setError('');

    try {
      console.log('Sending request to /api/auth/register');
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      console.log('Response status:', response.status);
      const responseData = await response.json();
      console.log('Response data:', responseData);

      if (!response.ok) {
        throw new Error(responseData.error || responseData.message || 'Something went wrong');
      }

      const { session } = responseData;
      
      if (session) {
        // Set the session in Supabase client
        await supabase.auth.setSession({
          access_token: session.access_token,
          refresh_token: session.refresh_token,
        });

        toast.success('Account created successfully!');
        router.push('/dashboard');
      } else {
        // If no session is returned, try to sign in
        const { data: signInData, error: signInError } = await supabase.auth.signInWithPassword({
          email: data.email,
          password: data.password,
        });

        if (signInError) {
          throw new Error(signInError.message);
        }

        if (signInData.session) {
          toast.success('Account created successfully!');
          router.push('/dashboard');
        } else {
          throw new Error('Failed to sign in after registration');
        }
      }
    } catch (err) {
      console.error('Registration error:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };

  const nextStep = () => {
    setCurrentStep(currentStep + 1);
  };

  const prevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-6 relative z-10">
      <div className="space-y-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Create your account</h1>
          <p className="text-gray-600 mt-3 text-lg">
            Join our community and start building your professional network
          </p>
        </div>

        {error && (
          <Alert variant="destructive" className="animate-in fade-in-50 duration-300">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="flex justify-center mb-8">
          <div className="flex space-x-2">
            {[1, 2].map((step) => (
              <div 
                key={step}
                className={`h-2 w-12 rounded-full transition-all duration-300 ${
                  currentStep >= step ? 'bg-blue-500' : 'bg-gray-200'
                }`}
              />
            ))}
          </div>
        </div>

        <FormProvider {...methods}>
          <form onSubmit={methods.handleSubmit(onSubmit)} className="space-y-8 relative">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="relative z-20"
            >
              {currentStep === 1 && <BasicInfoStep />}
              {currentStep === 2 && <PreferencesStep />}
            </motion.div>

            <div className="flex justify-between items-center pt-6 relative z-20">
              {currentStep > 1 && (
                <Button
                  type="button"
                  variant="outline"
                  onClick={prevStep}
                  className="px-6 py-2 rounded-full"
                >
                  Back
                </Button>
              )}
              <div className="ml-auto">
                {currentStep < 2 ? (
                  <Button
                    type="button"
                    onClick={nextStep}
                    className="px-8 py-2 rounded-full bg-blue-500 hover:bg-blue-600 text-white"
                  >
                    Continue
                  </Button>
                ) : (
                  <Button 
                    type="submit" 
                    className="px-8 py-2 rounded-full bg-blue-500 hover:bg-blue-600 text-white"
                    disabled={isSubmitting}
                  >
                    {isSubmitting ? 'Creating account...' : 'Create Account'}
                  </Button>
                )}
              </div>
            </div>
          </form>
        </FormProvider>
      </div>
    </div>
  );
} 