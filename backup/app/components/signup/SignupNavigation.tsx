'use client';

import { Button } from '../../../components/ui/button';
import { SignupFormData, SignupStep } from '../../types/signup';
import { motion } from 'framer-motion';

interface SignupNavigationProps {
  currentStep: SignupStep;
  onBack: () => void;
  isSubmitting: boolean;
}

export function SignupNavigation({
  currentStep,
  onBack,
  isSubmitting,
}: SignupNavigationProps) {
  const isLastStep = currentStep === 'summary';

  return (
    <motion.div 
      className="flex justify-between mt-10"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <Button
          variant="outline"
          onClick={onBack}
          disabled={currentStep === 'basic-info' || isSubmitting}
          className="px-6 py-2.5 text-sm font-medium rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue transition-all duration-200"
        >
          Back
        </Button>
      </motion.div>
      
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <Button
          type="submit"
          disabled={isSubmitting}
          className="px-6 py-2.5 text-sm font-medium rounded-lg bg-connection-blue text-white hover:bg-connection-blue-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue transition-all duration-200"
        >
          {isSubmitting ? (
            <span className="flex items-center">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </span>
          ) : isLastStep ? 'Complete Signup' : 'Continue'}
        </Button>
      </motion.div>
    </motion.div>
  );
} 