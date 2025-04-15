import React from 'react';
import * as SelectPrimitive from '@radix-ui/react-select';
import { cn } from '@/lib/utils';

interface Option {
  label: string;
  value: string;
}

interface SelectProps {
  value: string | string[];
  onChange: (value: string | string[]) => void;
  options: Option[];
  multiple?: boolean;
  className?: string;
}

export function Select({ value, onChange, options, multiple, className }: SelectProps) {
  if (multiple) {
    return (
      <div className={cn("relative", className)}>
        <select
          multiple
          value={value as string[]}
          onChange={(e) => {
            const values = Array.from(e.target.selectedOptions).map(opt => opt.value);
            onChange(values);
          }}
          className="w-full rounded-md border border-gray-300 bg-white py-2 pl-3 pr-10 text-sm"
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <SelectPrimitive.Root value={value as string} onValueChange={onChange as (value: string) => void}>
      <SelectPrimitive.Trigger className={cn(
        "inline-flex items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm",
        className
      )}>
        <SelectPrimitive.Value />
        <SelectPrimitive.Icon className="ml-2" />
      </SelectPrimitive.Trigger>

      <SelectPrimitive.Portal>
        <SelectPrimitive.Content className="overflow-hidden rounded-md border border-gray-300 bg-white shadow-md">
          <SelectPrimitive.Viewport className="p-1">
            {options.map((option) => (
              <SelectPrimitive.Item
                key={option.value}
                value={option.value}
                className="relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-gray-100"
              >
                <SelectPrimitive.ItemText>{option.label}</SelectPrimitive.ItemText>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
} 