import { useEffect, useState } from "react";
import { useRouter } from "next/router";
import { useAuthContext } from "@/providers/AuthProvider";
import AlignedMembersList from "@/components/AlignedMembersList";
import { Button } from "@/components/ui/button";
import { ChevronLeftIcon } from "@heroicons/react/24/outline";
import Link from "next/link";
import { Loader2 } from "lucide-react";

export default function GroupMemberAlignmentPage() {
  const router = useRouter();
  const { id: groupId } = router.query;
  const { user, loading } = useAuthContext();
  const [groupName, setGroupName] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  // Fetch group details to display the name
  useEffect(() => {
    if (!groupId || !user?.id) return;

    const fetchGroup = async () => {
      try {
        const response = await fetch(`/api/groups/${groupId}`);
        if (response.ok) {
          const data = await response.json();
          setGroupName(data.name || "Group");
        }
      } catch (error) {
        console.error("Error fetching group details:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchGroup();
  }, [groupId, user?.id]);

  if (loading || isLoading) {
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

  if (!user?.id || !groupId) {
    router.push(`/groups/${groupId}`);
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
          <Link href={`/groups/${groupId}`}>
            <ChevronLeftIcon className="h-4 w-4 mr-1" />
            Back to {groupName}
          </Link>
        </Button>
      </div>
      
      <h1 className="text-2xl font-bold mb-6">
        Connect with members in {groupName}
      </h1>
      
      <div className="prose mb-8 max-w-none">
        <p>
          Now that you've joined this group, here are some members you might want to connect with
          based on your professional profile, skills, and interests.
        </p>
      </div>
      
      <AlignedMembersList
        userId={user.id}
        entityType="group"
        entityId={groupId as string}
      />
      
      <div className="mt-10 pt-6 border-t">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium">Ready to participate?</h2>
          <Button asChild>
            <Link href={`/groups/${groupId}/discussions`}>View Discussions</Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 