'use client';

import { useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { motion, AnimatePresence } from 'framer-motion';
import { SignupFormData, SignupStep, SIGNUP_STEPS } from '../../types/signup';
import { BasicInfoStep } from './BasicInfoStep';
import { ProfessionalInfoStep } from './ProfessionalInfoStep';
import { PreferencesStep } from './PreferencesStep';
import { SocialLinksStep } from './SocialLinksStep';
import { SummaryStep } from './SummaryStep';
import { SignupProgress } from './SignupProgress';
import { SignupNavigation } from './SignupNavigation';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import Image from 'next/image';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

const stepComponents = {
  'basic-info': BasicInfoStep,
  'professional-info': ProfessionalInfoStep,
  'preferences': PreferencesStep,
  'social-links': SocialLinksStep,
  'summary': SummaryStep,
};

export function SignupFlow() {
  const [currentStep, setCurrentStep] = useState<SignupStep>('basic-info');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();
  const supabase = createClientComponentClient();

  const methods = useForm<SignupFormData>({
    mode: 'onChange',
    defaultValues: {
      firstName: '',
      lastName: '',
      email: '',
      password: '',
      confirmPassword: '',
      zipCode: '',
      title: '',
      company: '',
      industry: '',
      experience: '',
      skills: [],
      professionalInterests: [],
      bio: '',
      expertise: '',
      needs: '',
      meaningfulGoal: '',
      termsAccepted: false,
      values: [],
      goals: [],
      interests: [],
      networkingStyle: [],
      linkedin: '',
      github: '',
      portfolio: '',
      twitter: '',
      profileVisibility: 'public',
      emailNotifications: true,
      marketingEmails: false,
    },
  });

  const handleNext = () => {
    const currentIndex = SIGNUP_STEPS.findIndex(step => step.id === currentStep);
    if (currentIndex < SIGNUP_STEPS.length - 1) {
      setCurrentStep(SIGNUP_STEPS[currentIndex + 1].id);
    }
  };

  const handleBack = () => {
    const currentIndex = SIGNUP_STEPS.findIndex(step => step.id === currentStep);
    if (currentIndex > 0) {
      setCurrentStep(SIGNUP_STEPS[currentIndex - 1].id);
    }
  };

  const handleSubmit = async (data: SignupFormData) => {
    if (currentStep !== 'summary') {
      // If not on the final step, just move to the next step
      handleNext();
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Registration failed');
      }

      if (result.user) {
        toast.success('Account created successfully!');
        router.push('/dashboard');
      } else {
        throw new Error('Registration failed - no user returned');
      }
    } catch (error) {
      console.error('Registration error:', error);
      methods.setError('root', {
        type: 'manual',
        message: error instanceof Error ? error.message : 'Registration failed',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const CurrentStepComponent = stepComponents[currentStep];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-3xl">
        <div className="text-center mb-8">
          <Image
            src="/logos/networkli-logo-blue.png"
            alt="Networkli Logo"
            width={48}
            height={48}
            className="mx-auto"
          />
          <h2 className="mt-6 text-3xl font-bold tracking-tight text-gray-900">
            Create your account
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            Join our professional network and start connecting
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10"
        >
          <FormProvider {...methods}>
            <form onSubmit={methods.handleSubmit(handleSubmit)} className="space-y-8 sm:space-y-10">
              <div className="overflow-x-auto -mx-6 sm:mx-0 px-6 sm:px-0">
                <div className="min-w-[600px] sm:min-w-0">
                  <SignupProgress currentStep={currentStep} steps={SIGNUP_STEPS} formData={methods.watch()} />
                </div>
              </div>
              
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentStep}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="bg-white rounded-xl"
                >
                  <CurrentStepComponent />
                </motion.div>
              </AnimatePresence>

              <SignupNavigation
                currentStep={currentStep}
                onBack={handleBack}
                isSubmitting={isSubmitting}
              />
            </form>
          </FormProvider>
        </motion.div>
      </div>
    </div>
  );
} 