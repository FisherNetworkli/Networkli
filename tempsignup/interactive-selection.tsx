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
    <div className="w-full" ref={containerRef}>
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
          "flex min-h-[38px] w-full flex-wrap gap-1 rounded-md border border-gray-300 px-3 py-2 text-sm ring-offset-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 cursor-pointer",
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
                className="flex items-center gap-1 rounded-md bg-primary/10 px-2 py-1 text-xs"
              >
                {option.label}
                <button
                  className="ml-1 rounded-full p-0.5 hover:bg-primary/20"
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
              <span className="text-xs text-muted-foreground ml-1">
                {value.length}/{maxSelected} selected
              </span>
            )}
          </>
        ) : (
          <span className="text-muted-foreground">{placeholder}</span>
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
            className={cn(
              "absolute z-50 mt-1 w-full rounded-md border shadow-md",
              className
            )}
          >
            {/* Search input */}
            <div className="p-2 border-b">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder={searchPlaceholder}
                  className={cn(
                    "w-full rounded-md border border-input pl-8 pr-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                    className
                  )}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {/* Categories tabs */}
            {categories.length > 1 && (
              <div className="flex overflow-x-auto border-b p-1">
                <button
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium rounded-md mr-1",
                    selectedCategory === null
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-muted"
                  )}
                  onClick={() => setSelectedCategory(null)}
                >
                  All
                </button>
                {categories.map((category) => (
                  <button
                    key={category}
                    className={cn(
                      "px-3 py-1.5 text-xs font-medium rounded-md mr-1",
                      selectedCategory === category
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-muted"
                    )}
                    onClick={() => setSelectedCategory(category)}
                  >
                    {category}
                  </button>
                ))}
              </div>
            )}

            {/* Options list */}
            <div className="max-h-[300px] overflow-y-auto p-2">
              {Object.entries(filteredOptions).map(([category, categoryOptions]) => {
                // Skip if category is selected and doesn't match
                if (selectedCategory && selectedCategory !== category) {
                  return null;
                }

                return (
                  <div key={category} className="mb-4 last:mb-0">
                    {categories.length > 1 && (
                      <h4 className="text-xs font-medium text-muted-foreground mb-2 px-1">
                        {category}
                      </h4>
                    )}
                    <div className="grid grid-cols-1 gap-1">
                      {categoryOptions.map((option) => {
                        const isSelected = value.includes(option.value);
                        return (
                          <motion.div
                            key={option.value}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            className={cn(
                              "flex items-center justify-between p-2 rounded-md cursor-pointer",
                              isSelected
                                ? "bg-primary/10 border border-primary/20"
                                : "hover:bg-muted border border-transparent"
                            )}
                            onClick={() => handleSelect(option.value)}
                          >
                            <div className="flex items-center gap-2">
                              {option.icon && (
                                <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-full bg-primary/10">
                                  <span className="text-xs">{option.icon}</span>
                                </div>
                              )}
                              <div>
                                <div className="text-sm font-medium">{option.label}</div>
                                {option.description && (
                                  <div className="text-xs text-muted-foreground">
                                    {option.description}
                                  </div>
                                )}
                              </div>
                            </div>
                            <div
                              className={cn(
                                "w-5 h-5 rounded-sm border flex items-center justify-center",
                                isSelected
                                  ? "border-primary bg-primary text-primary-foreground"
                                  : "border-input"
                              )}
                            >
                              {isSelected && <Check className="h-3 w-3" />}
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
} 