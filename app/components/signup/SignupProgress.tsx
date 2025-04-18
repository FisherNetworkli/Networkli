'use client';

import { StepConfig, SignupStep, SignupFormData } from '../../types/signup';
import { Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface SignupProgressProps {
  steps: StepConfig[];
  currentStep: SignupStep;
  formData: Partial<SignupFormData>;
}

export function SignupProgress({ steps, currentStep, formData }: SignupProgressProps) {
  return (
    <div className="relative mb-12 sm:mb-16">
      <div className="absolute inset-0 flex items-center" aria-hidden="true">
        <div className="h-0.5 w-full bg-gray-200" />
      </div>
      <div className="relative flex justify-between">
        {steps.map((step, index) => {
          const isComplete = step.isComplete(formData);
          const isCurrent = step.id === currentStep;
          const isPast = steps.findIndex(s => s.id === currentStep) > index;

          return (
            <motion.div 
              key={step.id} 
              className="flex flex-col items-center"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <div
                className={cn(
                  'relative flex h-6 w-6 sm:h-8 sm:w-8 items-center justify-center rounded-full border-2 transition-all duration-300',
                  isComplete || isPast
                    ? 'border-connection-blue bg-connection-blue text-white'
                    : isCurrent
                    ? 'border-connection-blue bg-white text-connection-blue'
                    : 'border-gray-300 bg-white text-gray-300'
                )}
              >
                {isComplete ? (
                  <Check className="h-3 w-3 sm:h-4 sm:w-4" />
                ) : (
                  <span className="text-[10px] sm:text-xs font-medium">{index + 1}</span>
                )}
              </div>
              <div className="mt-2 text-center w-[80px] sm:w-[120px]">
                <span
                  className={cn(
                    'text-xs sm:text-sm font-medium transition-colors duration-200',
                    isComplete || isPast || isCurrent
                      ? 'text-gray-900'
                      : 'text-gray-400'
                  )}
                >
                  {step.title}
                </span>
                <p className="mt-1 text-[10px] sm:text-xs text-gray-400 leading-tight hidden sm:block">
                  {step.description}
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
} 