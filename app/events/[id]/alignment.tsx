import { useEffect, useState } from "react";
import { useRouter } from "next/router";
import { useAuthUser } from "@/app/hooks/useAuthUser";
import AlignedMembersList from "@/app/components/AlignedMembersList";
import { Button } from "@/app/components/ui/button";
import { ChevronLeftIcon } from "@heroicons/react/24/outline";
import Link from "next/link";

export default function EventMemberAlignmentPage() {
  const router = useRouter();
  const { id: eventId } = router.query;
  const { user, isLoading: userLoading } = useAuthUser();
  const [eventName, setEventName] = useState("");
  const [isLoading, setIsLoading] = useState(true);

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

  if (userLoading || isLoading) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-8">
        <div className="animate-pulse">
          <div className="h-6 w-48 bg-gray-200 rounded mb-4"></div>
          <div className="h-10 w-96 bg-gray-200 rounded mb-6"></div>
          <div className="h-64 bg-gray-100 rounded"></div>
        </div>
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
      
      <AlignedMembersList
        userId={user.id}
        entityType="event"
        entityId={eventId as string}
        title="Recommended connections at this event"
        initialLimit={5}
        maxLimit={15}
      />
      
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