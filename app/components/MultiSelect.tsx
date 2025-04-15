import React, { useState } from 'react';

interface Option {
  value: string;
  label: string;
  category?: string;
}

interface MultiSelectProps {
  options: Option[];
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
  label: string;
  required?: boolean;
  maxItems?: number;
}

export default function MultiSelect({
  options,
  value,
  onChange,
  placeholder = 'Select options...',
  label,
  required = false,
  maxItems,
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');

  const filteredOptions = options.filter(
    option =>
      !value.includes(option.value) &&
      (option.label.toLowerCase().includes(search.toLowerCase()) ||
        option.category?.toLowerCase().includes(search.toLowerCase()))
  );

  const handleSelect = (optionValue: string) => {
    if (maxItems && value.length >= maxItems) return;
    onChange([...value, optionValue]);
    setSearch('');
  };

  const handleRemove = (optionValue: string) => {
    onChange(value.filter(v => v !== optionValue));
  };

  const getOptionLabel = (optionValue: string) => {
    return options.find(o => o.value === optionValue)?.label || optionValue;
  };

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      <div
        className="min-h-[42px] p-1.5 border border-gray-300 rounded-md cursor-text"
        onClick={() => setIsOpen(true)}
      >
        <div className="flex flex-wrap gap-1.5">
          {value.map(v => (
            <span
              key={v}
              className="inline-flex items-center px-2.5 py-0.5 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
            >
              {getOptionLabel(v)}
              <button
                type="button"
                onClick={e => {
                  e.stopPropagation();
                  handleRemove(v);
                }}
                className="ml-1.5 inline-flex items-center justify-center w-4 h-4 rounded-full hover:bg-blue-200"
              >
                ×
              </button>
            </span>
          ))}
          {(!maxItems || value.length < maxItems) && (
            <input
              type="text"
              className="outline-none border-none bg-transparent flex-1 min-w-[120px]"
              placeholder={value.length === 0 ? placeholder : ''}
              value={search}
              onChange={e => setSearch(e.target.value)}
              onFocus={() => setIsOpen(true)}
            />
          )}
        </div>
      </div>
      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          ></div>
          <ul className="absolute z-20 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
            {filteredOptions.length === 0 ? (
              <li className="px-4 py-2 text-sm text-gray-500">No options found</li>
            ) : (
              filteredOptions.map(option => (
                <li
                  key={option.value}
                  onClick={() => handleSelect(option.value)}
                  className="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer flex items-center justify-between"
                >
                  <span>{option.label}</span>
                  {option.category && (
                    <span className="text-xs text-gray-500">{option.category}</span>
                  )}
                </li>
              ))
            )}
          </ul>
        </>
      )}
    </div>
  );
} 