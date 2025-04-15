import { Metadata } from 'next';
import { SignupFlow } from '../components/signup/SignupFlow';

export const metadata: Metadata = {
  title: 'Sign Up | Networkli',
  description: 'Create your Networkli account and start connecting with professionals',
};

export default function SignUpPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <SignupFlow />
    </div>
  );
}