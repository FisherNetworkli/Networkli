import Image from 'next/image';
import Link from 'next/link';

export default function SmartMatchingPage() {
  return (
    <div className="min-h-screen bg-white">
      <main className="pt-16">
        <section className="relative h-[40vh] min-h-[300px]">
          <Image
            src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images/20250412_1521_Cozy%20Security%20Workspace_simple_compose_01jrnxz6raeagvgh3esfn2x3v0.png"
            alt="Smart Matching"
            fill
            className="object-cover"
            sizes="(max-width: 768px) 100vw, 50vw"
          />
          <div className="absolute inset-0 bg-black/30" />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-white">
              <h1 className="text-4xl md:text-5xl font-bold mb-4">Smart Matching</h1>
              <p className="text-xl md:text-2xl">Find your perfect professional match</p>
            </div>
          </div>
        </section>

        <section className="py-16 px-4 md:px-8 max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-6">AI-Powered Matching</h2>
              <p className="text-lg text-gray-600 mb-6">
                Our advanced matching algorithm considers your skills, interests, and professional goals
                to connect you with the most relevant professionals in your field.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Skill-based matching</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Interest alignment</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Career goal compatibility</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Industry-specific connections</span>
                </li>
              </ul>
            </div>
            <div className="relative h-[400px]">
              <Image
                src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images/20250412_1525_Cafe%20Connection%20Aligned_simple_compose_01jrny7phzeakvaw3xff9jn2aq.png"
                alt="Smart Matching Features"
                fill
                className="object-cover rounded-lg"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </div>
          </div>
        </section>

        <section className="bg-gray-50 py-16 px-4 md:px-8">
          <div className="max-w-7xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="bg-primary/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">1</span>
                </div>
                <h3 className="text-xl font-semibold mb-2">Profile Analysis</h3>
                <p className="text-gray-600">We analyze your professional profile and preferences</p>
              </div>
              <div className="text-center">
                <div className="bg-primary/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">2</span>
                </div>
                <h3 className="text-xl font-semibold mb-2">Smart Matching</h3>
                <p className="text-gray-600">Our AI finds the best matches for you</p>
              </div>
              <div className="text-center">
                <div className="bg-primary/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">3</span>
                </div>
                <h3 className="text-xl font-semibold mb-2">Connect & Grow</h3>
                <p className="text-gray-600">Start building meaningful professional relationships</p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 px-4 md:px-8 max-w-7xl mx-auto">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to Find Your Match?</h2>
            <p className="text-lg text-gray-600 mb-8">
              Join our community and discover your perfect professional connections.
            </p>
            <Link
              href="/signup"
              className="inline-block bg-primary text-white px-8 py-3 rounded-lg font-semibold hover:bg-primary/90 transition-colors"
            >
              Get Started
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
} 