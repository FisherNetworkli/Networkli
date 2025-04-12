'use client';

import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../components/ui/card";
import { Check } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function SubscriptionPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null);

  const handleUpgrade = async (plan: string) => {
    try {
      setLoading(true);
      setSelectedPlan(plan);
      
      const response = await fetch('/api/create-checkout-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ plan }),
      });
      
      const data = await response.json();
      
      if (data.url) {
        router.push(data.url);
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
      setSelectedPlan(null);
    }
  };

  return (
    <div className="container max-w-6xl py-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-4">Choose Your Plan</h1>
        <p className="text-muted-foreground text-lg">
          Get access to premium features and unlock your full potential
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        {/* Free Plan */}
        <Card>
          <CardHeader>
            <CardTitle>Free</CardTitle>
            <CardDescription>Basic features for getting started</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-4">$0<span className="text-lg font-normal text-muted-foreground">/month</span></div>
            <ul className="space-y-2">
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Basic profile
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Limited connections
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Basic search
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Community access
              </li>
            </ul>
          </CardContent>
          <CardFooter>
            <Button className="w-full bg-gray-100 text-gray-900 hover:bg-gray-200">Current Plan</Button>
          </CardFooter>
        </Card>

        {/* Premium Plan */}
        <Card className="border-primary">
          <CardHeader>
            <CardTitle>Premium</CardTitle>
            <CardDescription>Everything you need to succeed</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-4">$12.99<span className="text-lg font-normal text-muted-foreground">/month</span></div>
            <ul className="space-y-2">
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Advanced profile features
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Unlimited connections
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Priority search results
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Analytics dashboard
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Premium support
              </li>
            </ul>
          </CardContent>
          <CardFooter>
            <Button 
              className="w-full" 
              onClick={() => handleUpgrade('premium')}
              disabled={loading && selectedPlan === 'premium'}
            >
              {loading && selectedPlan === 'premium' ? 'Loading...' : 'Upgrade to Premium'}
            </Button>
          </CardFooter>
        </Card>

        {/* Organizer Plan */}
        <Card className="border-primary border-2">
          <CardHeader>
            <CardTitle>Organizer</CardTitle>
            <CardDescription>For creating groups and events</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-4">$99<span className="text-lg font-normal text-muted-foreground">/month</span></div>
            <ul className="space-y-2">
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                All Premium features
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Create unlimited groups
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Host events and meetups
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Advanced analytics
              </li>
              <li className="flex items-center">
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Priority support
              </li>
            </ul>
          </CardContent>
          <CardFooter>
            <Button 
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700" 
              onClick={() => handleUpgrade('organizer')}
              disabled={loading && selectedPlan === 'organizer'}
            >
              {loading && selectedPlan === 'organizer' ? 'Loading...' : 'Contact for Organizer Plan'}
            </Button>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
} 