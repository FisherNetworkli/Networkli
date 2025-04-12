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

  const methods = useForm<SignupFormData>({
    mode: 'onChange',
    defaultValues: {
      firstName: '',
      lastName: '',
      email: '',
      password: '',
      confirmPassword: '',
      title: '',
      company: '',
      industry: '',
      experience: '',
      skills: [],
      bio: '',
      interests: [],
      lookingFor: [],
      preferredIndustries: [],
      preferredRoles: [],
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
      handleNext();
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error('Signup failed');
      }

      toast.success('Account created successfully!');
      router.push('/dashboard');
    } catch (error) {
      toast.error('Failed to create account. Please try again.');
      console.error('Signup error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const CurrentStepComponent = stepComponents[currentStep];

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <div className="flex-grow flex flex-col items-center justify-center px-4 sm:px-6 lg:px-8 py-8">
        <motion.div 
          className="w-full max-w-2xl bg-white rounded-2xl shadow-sm px-6 sm:px-12 py-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex flex-col items-center mb-12">
            <h1 className="text-2xl sm:text-3xl font-semibold text-gray-900 tracking-tight mb-2">
              Join Networkli
            </h1>
            <p className="text-lg text-gray-600 text-center max-w-md mb-8">
              Create your account and start building meaningful professional connections
            </p>
            <div className="mb-8">
              <Image 
                src="/logos/networkli-logo-blue.png"
                alt="Networkli Logo" 
                width={180}
                height={48} 
                className="w-auto h-10 sm:h-12"
                priority
              />
            </div>
            <div className="text-center">
              <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 tracking-tight mb-1">
                Create your account
              </h2>
              <p className="text-sm text-gray-500">
                Join our community of professionals
              </p>
            </div>
          </div>
          
          <FormProvider {...methods}>
            <form onSubmit={methods.handleSubmit(handleSubmit)} className="space-y-8 sm:space-y-10">
              <div className="overflow-x-auto -mx-6 sm:mx-0 px-6 sm:px-0">
                <div className="min-w-[600px] sm:min-w-0">
                  <SignupProgress currentStep={currentStep} steps={SIGNUP_STEPS} formData={methods.getValues()} />
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