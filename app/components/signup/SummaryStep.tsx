'use client';

import { useFormContext } from 'react-hook-form';
import { SignupFormData } from '@/app/types/signup';
import { Card } from '../../components/ui/card';
import { Badge } from '../../components/ui/badge';
import { 
  User, 
  Briefcase, 
  Heart, 
  Link
} from 'lucide-react';

type Field = {
  label: string;
  value: string | undefined;
  isList?: boolean;
};

export function SummaryStep() {
  const { watch } = useFormContext<SignupFormData>();
  const formData = watch();

  const sections = [
    {
      title: 'Basic Information',
      icon: User,
      fields: [
        { label: 'Name', value: `${formData.firstName} ${formData.lastName}` },
        { label: 'Email', value: formData.email },
      ],
    },
    {
      title: 'Professional Details',
      icon: Briefcase,
      fields: [
        { label: 'Title', value: formData.title },
        { label: 'Company', value: formData.company },
        { label: 'Industry', value: formData.industry },
        { label: 'Experience', value: formData.experience },
        { 
          label: 'Skills', 
          value: formData.skills?.join(', '),
          isList: true 
        },
      ],
    },
    {
      title: 'Preferences',
      icon: Heart,
      fields: [
        { 
          label: 'Interests', 
          value: formData.interests?.join(', '),
          isList: true 
        },
        { 
          label: 'Looking For', 
          value: formData.lookingFor?.join(', '),
          isList: true 
        },
        { 
          label: 'Preferred Industries', 
          value: formData.preferredIndustries?.join(', '),
          isList: true 
        },
      ],
    },
    {
      title: 'Social Links',
      icon: Link,
      fields: [
        { label: 'LinkedIn', value: formData.linkedin },
        { label: 'GitHub', value: formData.github },
        { label: 'Portfolio', value: formData.portfolio },
        { label: 'Twitter', value: formData.twitter },
      ].filter((field): field is Field => field.value !== undefined),
    },
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Review Your Information</h2>
      <p className="text-gray-500">
        Please review your information before completing your registration.
      </p>

      <div className="grid gap-6">
        {sections.map((section) => (
          <Card key={section.title} className="p-6">
            <div className="flex items-center gap-2 mb-4">
              <section.icon className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-medium">{section.title}</h3>
            </div>
            <div className="space-y-4">
              {section.fields.map((field) => (
                <div key={field.label} className="flex flex-col gap-1">
                  <span className="text-sm text-gray-500">{field.label}</span>
                  {field.isList ? (
                    <div className="flex flex-wrap gap-2">
                      {field.value?.split(', ').map((item) => (
                        <Badge key={item} variant="secondary">
                          {item}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{field.value || 'Not provided'}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
} 