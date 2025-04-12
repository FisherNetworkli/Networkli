'use client';

import React from 'react';
import { cn } from '@/lib/utils';

interface Option {
  value: string;
  label: string;
  description?: string;
}

interface InteractiveSelectionProps {
  options: Option[];
  selectedValues: string[];
  onChange: (values: string[]) => void;
  maxSelections?: number;
  className?: string;
}

export function InteractiveSelection({
  options,
  selectedValues,
  onChange,
  maxSelections = Infinity,
  className,
}: InteractiveSelectionProps) {
  const handleToggle = (value: string) => {
    if (selectedValues.includes(value)) {
      onChange(selectedValues.filter((v) => v !== value));
    } else if (selectedValues.length < maxSelections) {
      onChange([...selectedValues, value]);
    }
  };

  return (
    <div className={cn('grid grid-cols-1 md:grid-cols-2 gap-4', className)}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => handleToggle(option.value)}
          className={cn(
            'p-4 rounded-lg border text-left transition-colors',
            'hover:border-primary hover:bg-primary/5',
            'focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
            selectedValues.includes(option.value)
              ? 'border-primary bg-primary/10'
              : 'border-gray-200 bg-white',
            selectedValues.length >= maxSelections &&
              !selectedValues.includes(option.value) &&
              'opacity-50 cursor-not-allowed'
          )}
          disabled={
            selectedValues.length >= maxSelections &&
            !selectedValues.includes(option.value)
          }
        >
          <div className="flex items-center justify-between">
            <h3 className="font-medium">{option.label}</h3>
            {selectedValues.includes(option.value) && (
              <svg
                className="h-5 w-5 text-primary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            )}
          </div>
          {option.description && (
            <p className="mt-1 text-sm text-gray-500">{option.description}</p>
          )}
        </button>
      ))}
    </div>
  );
} 