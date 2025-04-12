'use client';

import { useFormContext } from 'react-hook-form';
import { SignupFormData } from '@/app/types/signup';
import { Label } from '../ui/label';
import { RadioGroup, RadioGroupItem } from '../ui/radio-group';
import { Switch } from '../ui/switch';
import { Card } from '../ui/card';
import { Globe, Lock, Users } from 'lucide-react';

export function ProfileOptionsStep() {
  const { register, watch } = useFormContext<SignupFormData>();
  const profileVisibility = watch('profileVisibility');

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Profile Settings</h2>
      <p className="text-gray-500">
        Customize how your profile appears and how you receive notifications.
      </p>

      <Card className="p-6">
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-medium mb-2">Profile Visibility</h3>
            <p className="text-sm text-gray-500 mb-4">
              Control who can see your profile information
            </p>
            <RadioGroup
              defaultValue="public"
              className="grid grid-cols-1 md:grid-cols-3 gap-4"
              onValueChange={(value) => {
                register('profileVisibility').onChange({
                  target: { value, name: 'profileVisibility' },
                });
              }}
            >
              <div className="flex items-center space-x-2 border rounded-md p-4 cursor-pointer hover:bg-gray-50">
                <RadioGroupItem value="public" id="public" />
                <div className="flex items-center">
                  <Globe className="h-5 w-5 mr-2 text-primary" />
                  <Label htmlFor="public" className="cursor-pointer">
                    <div className="font-medium">Public</div>
                    <div className="text-xs text-gray-500">Anyone can view your profile</div>
                  </Label>
                </div>
              </div>
              <div className="flex items-center space-x-2 border rounded-md p-4 cursor-pointer hover:bg-gray-50">
                <RadioGroupItem value="connections" id="connections" />
                <div className="flex items-center">
                  <Users className="h-5 w-5 mr-2 text-primary" />
                  <Label htmlFor="connections" className="cursor-pointer">
                    <div className="font-medium">Connections</div>
                    <div className="text-xs text-gray-500">Only your connections can view</div>
                  </Label>
                </div>
              </div>
              <div className="flex items-center space-x-2 border rounded-md p-4 cursor-pointer hover:bg-gray-50">
                <RadioGroupItem value="private" id="private" />
                <div className="flex items-center">
                  <Lock className="h-5 w-5 mr-2 text-primary" />
                  <Label htmlFor="private" className="cursor-pointer">
                    <div className="font-medium">Private</div>
                    <div className="text-xs text-gray-500">Only you can view your profile</div>
                  </Label>
                </div>
              </div>
            </RadioGroup>
          </div>

          <div className="pt-4 border-t">
            <h3 className="text-lg font-medium mb-2">Email Notifications</h3>
            <p className="text-sm text-gray-500 mb-4">
              Manage how you receive notifications
            </p>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="emailNotifications" className="font-medium">
                    Email Notifications
                  </Label>
                  <p className="text-sm text-gray-500">
                    Receive notifications about your account activity
                  </p>
                </div>
                <Switch
                  id="emailNotifications"
                  defaultChecked
                  onCheckedChange={(checked) => {
                    register('emailNotifications').onChange({
                      target: { checked, name: 'emailNotifications' },
                    });
                  }}
                />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="marketingEmails" className="font-medium">
                    Marketing Emails
                  </Label>
                  <p className="text-sm text-gray-500">
                    Receive updates about new features and promotions
                  </p>
                </div>
                <Switch
                  id="marketingEmails"
                  onCheckedChange={(checked) => {
                    register('marketingEmails').onChange({
                      target: { checked, name: 'marketingEmails' },
                    });
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
} 