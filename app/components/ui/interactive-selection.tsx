'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, X, Search } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface Option {
  label: string;
  value: string;
  category?: string;
  icon?: string;
  description?: string;
}

export interface InteractiveSelectionProps {
  id?: string;
  options: Option[];
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  maxSelected?: number;
  groupBy?: (option: Option) => string;
  title?: string;
  description?: string;
  className?: string;
}

export function InteractiveSelection({
  id,
  options = [],
  value = [],
  onChange,
  placeholder = 'Select options...',
  searchPlaceholder = 'Search...',
  maxSelected,
  groupBy,
  title,
  description,
  className,
}: InteractiveSelectionProps) {
  const [searchQuery, setSearchQuery] = React.useState('');
  const [selectedCategory, setSelectedCategory] = React.useState<string | null>(null);
  const [isOpen, setIsOpen] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Group options by category
  const groupedOptions = React.useMemo(() => {
    if (!groupBy) {
      return { 'Options': options };
    }

    return options.reduce((acc, option) => {
      const category = groupBy(option) || 'Other';
      if (!acc[category]) acc[category] = [];
      acc[category].push(option);
      return acc;
    }, {} as Record<string, Option[]>);
  }, [options, groupBy]);

  // Filter options based on search query
  const filteredOptions = React.useMemo(() => {
    if (!searchQuery) {
      return groupedOptions;
    }

    const query = searchQuery.toLowerCase();
    const filtered: Record<string, Option[]> = {};

    Object.entries(groupedOptions).forEach(([category, categoryOptions]) => {
      const matchingOptions = categoryOptions.filter(
        option => 
          option.label.toLowerCase().includes(query) || 
          (option.description && option.description.toLowerCase().includes(query))
      );
      
      if (matchingOptions.length > 0) {
        filtered[category] = matchingOptions;
      }
    });

    return filtered;
  }, [groupedOptions, searchQuery]);

  // Get categories
  const categories = React.useMemo(() => {
    return Object.keys(filteredOptions);
  }, [filteredOptions]);

  // Handle option selection
  const handleSelect = (optionValue: string) => {
    if (maxSelected && value.length >= maxSelected && !value.includes(optionValue)) {
      return;
    }

    const newValue = value.includes(optionValue)
      ? value.filter(v => v !== optionValue)
      : [...value, optionValue];

    onChange(newValue);
    // Don't close dropdown after selection for multi-select
    if (maxSelected === 1) {
      setIsOpen(false);
    }
  };

  // Handle option removal
  const handleRemove = (optionValue: string) => {
    const newValue = value.filter(v => v !== optionValue);
    onChange(newValue);
  };

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Get selected options
  const selectedOptions = React.useMemo(() => {
    return options.filter(option => value.includes(option.value));
  }, [options, value]);

  return (
    <div className="relative w-full" ref={containerRef}>
      {title && (
        <div className="mb-2">
          <h3 className="text-sm font-medium text-gray-900">{title}</h3>
          {description && <p className="text-xs text-gray-500">{description}</p>}
        </div>
      )}

      {/* Selected options display */}
      <div 
        id={id}
        className={cn(
          "flex min-h-[38px] w-full flex-wrap gap-1 rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 cursor-pointer",
          className
        )}
        onClick={() => setIsOpen(!isOpen)}
      >
        {selectedOptions.length > 0 ? (
          <>
            {selectedOptions.map((option) => (
              <motion.div
                key={option.value}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-1 rounded-md bg-blue-100 px-2 py-1 text-xs text-blue-700"
              >
                {option.label}
                <button
                  className="ml-1 rounded-full p-0.5 hover:bg-blue-200"
                  onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    handleRemove(option.value);
                  }}
                >
                  <X className="h-3 w-3" />
                </button>
              </motion.div>
            ))}
            {maxSelected && (
              <span className="text-xs text-gray-500 ml-1">
                {value.length}/{maxSelected} selected
              </span>
            )}
          </>
        ) : (
          <span className="text-gray-500">{placeholder}</span>
        )}
      </div>

      {/* Dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="absolute z-50 mt-1 w-full rounded-md border bg-white shadow-lg"
          >
            {/* Search input */}
            <div className="p-2 border-b bg-white">
              <div className="relative">
                <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
                <input
                  type="text"
                  className="w-full rounded-md border border-gray-300 pl-8 pr-4 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  placeholder={searchPlaceholder}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {/* Options list */}
            <div className="max-h-60 overflow-y-auto p-2">
              {Object.entries(filteredOptions).map(([category, categoryOptions]) => (
                <div key={category} className="mb-4 last:mb-0">
                  <div className="mb-2 px-2 text-sm font-medium text-gray-500">
                    {category}
                  </div>
                  <div className="space-y-1">
                    {categoryOptions.map((option) => {
                      const isSelected = value.includes(option.value);
                      const hasReachedMax = typeof maxSelected === 'number' && value.length >= maxSelected;
                      const isDisabled = hasReachedMax && !isSelected;

                      return (
                        <button
                          key={option.value}
                          className={cn(
                            "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm transition-colors",
                            isSelected
                              ? "bg-blue-100 text-blue-700 hover:bg-blue-200 cursor-pointer"
                              : isDisabled
                              ? "opacity-50"
                              : "text-gray-900 hover:bg-gray-100 cursor-pointer"
                          )}
                          onClick={() => !isDisabled && handleSelect(option.value)}
                          type="button"
                        >
                          <div className="flex flex-col items-start">
                            <span className="font-medium">{option.label}</span>
                            {option.description && (
                              <span className="text-xs text-gray-500">{option.description}</span>
                            )}
                          </div>
                          {isSelected && <Check className="h-4 w-4 shrink-0" />}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
              {Object.keys(filteredOptions).length === 0 && (
                <div className="px-2 py-4 text-center text-sm text-gray-500">
                  No options found
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
} 