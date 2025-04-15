'use client';

import * as React from 'react';
import { Check, ChevronsUpDown, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from '@/components/ui/command';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Badge } from '@/components/ui/badge';

interface Option {
  value: string;
  label: string;
  category?: string;
}

interface InteractiveSelectionProps {
  id: string;
  value: string[];
  onChange: (value: string[]) => void;
  options: Option[];
  groupBy: (option: Option) => string;
  maxSelected?: number;
  placeholder?: string;
  searchPlaceholder?: string;
  title?: string;
  description?: string;
  className?: string;
}

export function InteractiveSelection({
  id,
  value,
  onChange,
  options,
  groupBy,
  maxSelected,
  placeholder = 'Select items...',
  searchPlaceholder = 'Search...',
  title,
  description,
  className,
}: InteractiveSelectionProps) {
  const [open, setOpen] = React.useState(false);

  const selectedItems = options.filter((option) => value.includes(option.value));
  const groupedOptions = options.reduce((acc, option) => {
    const group = groupBy(option);
    if (!acc[group]) {
      acc[group] = [];
    }
    acc[group].push(option);
    return acc;
  }, {} as Record<string, Option[]>);

  const handleSelect = (optionValue: string) => {
    if (value.includes(optionValue)) {
      onChange(value.filter((v) => v !== optionValue));
    } else if (!maxSelected || value.length < maxSelected) {
      onChange([...value, optionValue]);
    }
  };

  const handleRemove = (optionValue: string) => {
    onChange(value.filter((v) => v !== optionValue));
  };

  return (
    <div className={cn('flex flex-col gap-2', className)}>
      {(title || description) && (
        <div className="space-y-1">
          {title && <h3 className="text-sm font-medium">{title}</h3>}
          {description && (
            <p className="text-sm text-gray-500">{description}</p>
          )}
        </div>
      )}

      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between bg-white"
          >
            {selectedItems.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                {selectedItems.map((item) => (
                  <Badge
                    key={item.value}
                    variant="secondary"
                    className="mr-1"
                  >
                    {item.label}
                  </Badge>
                ))}
              </div>
            ) : (
              <span className="text-gray-500">{placeholder}</span>
            )}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-full p-0">
          <Command>
            <CommandInput placeholder={searchPlaceholder} />
            <CommandEmpty>No results found.</CommandEmpty>
            {Object.entries(groupedOptions).map(([group, items]) => (
              <CommandGroup key={group} heading={group}>
                {items.map((option) => (
                  <CommandItem
                    key={option.value}
                    value={option.value}
                    onSelect={() => handleSelect(option.value)}
                  >
                    <Check
                      className={cn(
                        'mr-2 h-4 w-4',
                        value.includes(option.value)
                          ? 'opacity-100'
                          : 'opacity-0'
                      )}
                    />
                    {option.label}
                  </CommandItem>
                ))}
              </CommandGroup>
            ))}
          </Command>
        </PopoverContent>
      </Popover>

      {maxSelected && (
        <p className="text-xs text-gray-500">
          {value.length}/{maxSelected} selected
        </p>
      )}
    </div>
  );
} 