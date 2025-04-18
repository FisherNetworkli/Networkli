import { useEffect, useState } from "react";
import { useRouter } from "next/router";
import { useAuthContext } from "@/providers/AuthProvider";
import AlignedMembersList from "@/components/AlignedMembersList";
import { Button } from "@/components/ui/button";
import { ChevronLeftIcon } from "@heroicons/react/24/outline";
import Link from "next/link";
import { Loader2 } from "lucide-react";

export default function EventMemberAlignmentPage() {
  const router = useRouter();
  const { id: eventId } = router.query;
  const { user, loading } = useAuthContext();
  const [eventName, setEventName] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch event details to display the name
  useEffect(() => {
    if (!eventId || !user?.id) return;

    const fetchEvent = async () => {
      try {
        const response = await fetch(`/api/events/${eventId}`);
        if (response.ok) {
          const data = await response.json();
          setEventName(data.title || "Event");
        }
      } catch (error) {
        console.error("Error fetching event details:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchEvent();
  }, [eventId, user?.id]);

  if (loading || isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-12 w-12 animate-spin text-primary" />
      </div>
    );
  }

  if (!user?.id || !eventId) {
    router.push(`/events/${eventId}`);
    return null;
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="mb-2">
        <Button
          variant="ghost"
          size="sm"
          asChild
          className="text-sm text-muted-foreground"
        >
          <Link href={`/events/${eventId}`}>
            <ChevronLeftIcon className="h-4 w-4 mr-1" />
            Back to {eventName}
          </Link>
        </Button>
      </div>
      
      <h1 className="text-2xl font-bold mb-6">
        Connect with attendees at {eventName}
      </h1>
      
      <div className="prose mb-8 max-w-none">
        <p>
          Based on your RSVP to this event, here are some attendees you might want to connect with
          based on your professional profile, skills, and interests.
        </p>
      </div>
      
      <main className="mt-8">
        {user && (
          <AlignedMembersList
            userId={user.id}
            entityType="event"
            entityId={eventId as string}
          />
        )}
      </main>
      
      <div className="mt-10 pt-6 border-t">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium">Event Details</h2>
          <Button asChild>
            <Link href={`/events/${eventId}`}>View Event Information</Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 