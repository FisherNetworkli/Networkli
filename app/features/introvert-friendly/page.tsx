import Image from 'next/image';
import Link from 'next/link';

export default function IntrovertFriendlyPage() {
  return (
    <div className="min-h-screen bg-white">
      <main className="pt-16">
        <section className="relative h-[40vh] min-h-[300px]">
          <Image
            src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images/20250412_1519_Cozy%20Networking%20Lounge_simple_compose_01jrnxwpvafr2vv3b404hr8h2c.png"
            alt="Introvert Friendly Networking"
            fill
            className="object-cover"
            sizes="(max-width: 768px) 100vw, 50vw"
          />
          <div className="absolute inset-0 bg-black/30" />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-white">
              <h1 className="text-4xl md:text-5xl font-bold mb-4">Introvert-Friendly Networking</h1>
              <p className="text-xl md:text-2xl">Connect at your own pace, in your comfort zone</p>
            </div>
          </div>
        </section>

        <section className="py-16 px-4 md:px-8 max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-6">Networking That Respects Your Space</h2>
              <p className="text-lg text-gray-600 mb-6">
                We understand that networking doesn't have to be overwhelming. Our platform is designed
                with introverts in mind, providing a comfortable space to build meaningful connections.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Connect at your own pace</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Meaningful one-on-one conversations</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>No pressure to attend large events</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">✓</span>
                  <span>Focus on quality over quantity</span>
                </li>
              </ul>
            </div>
            <div className="relative h-[400px]">
              <Image
                src="https://tmctlkjnjnirafxgfnza.supabase.co/storage/v1/object/public/images/20250412_1520_Perfect%20Match%20Found!_simple_compose_01jrnxy1f9ftkahscvm7fw5xj8.png"
                alt="Introvert Friendly Features"
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
                <h3 className="text-xl font-semibold mb-2">Create Your Profile</h3>
                <p className="text-gray-600">Set your preferences and comfort level for networking</p>
              </div>
              <div className="text-center">
                <div className="bg-primary/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">2</span>
                </div>
                <h3 className="text-xl font-semibold mb-2">Get Matched</h3>
                <p className="text-gray-600">Connect with like-minded professionals</p>
              </div>
              <div className="text-center">
                <div className="bg-primary/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">3</span>
                </div>
                <h3 className="text-xl font-semibold mb-2">Grow Naturally</h3>
                <p className="text-gray-600">Build relationships at your own pace</p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 px-4 md:px-8 max-w-7xl mx-auto">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to Start?</h2>
            <p className="text-lg text-gray-600 mb-8">
              Join our community of professionals who value meaningful connections.
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