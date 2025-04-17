import { Avatar, Button, Card } from "@/components/ui";
import { cn } from "@/lib/utils";
import { UserIcon } from "@heroicons/react/24/outline";
import Link from "next/link";

interface MemberAlignmentCardProps {
  member: {
    id: string;
    full_name: string;
    avatar_url?: string;
    title?: string;
    company?: string;
    similarity_score: number;
    alignment_reason?: string;
    [key: string]: any;
  };
  entityType: "group" | "event";
  entityId: string;
  onActionClick?: () => void;
  actionText?: string;
  showScore?: boolean;
  className?: string;
}

/**
 * A card component that displays a member with alignment information.
 * Used to show aligned members in groups or events.
 */
export function MemberAlignmentCard({
  member,
  entityType,
  entityId,
  onActionClick,
  actionText = "Connect",
  showScore = true,
  className,
}: MemberAlignmentCardProps) {
  // Format the similarity score as a percentage
  const scorePercent = Math.round(member.similarity_score * 100);
  
  // Log click interaction
  async function logAlignmentClick() {
    try {
      await fetch("/api/track-interaction", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          type: "MEMBER_ALIGNMENT_CLICK",
          targetId: member.id,
          metadata: {
            entityType,
            entityId,
            similarityScore: member.similarity_score
          }
        }),
      });
    } catch (error) {
      console.error("Failed to log interaction:", error);
    }
  }

  return (
    <Card className={cn("flex flex-col overflow-hidden", className)}>
      <div className="flex items-start p-4 gap-4">
        <Avatar
          src={member.avatar_url}
          fallback={<UserIcon className="h-6 w-6 text-muted-foreground" />}
          className="h-12 w-12 rounded-full border"
        />
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <Link 
              href={`/profile/${member.id}`} 
              className="font-medium hover:underline truncate"
              onClick={logAlignmentClick}
            >
              {member.full_name}
            </Link>
            
            {showScore && (
              <span className="text-xs font-medium text-muted-foreground bg-muted px-2 py-0.5 rounded-full whitespace-nowrap">
                {scorePercent}% match
              </span>
            )}
          </div>
          
          {member.title && (
            <p className="text-sm text-muted-foreground truncate">
              {member.title}{member.company ? ` at ${member.company}` : ""}
            </p>
          )}
          
          {member.alignment_reason && (
            <p className="text-sm mt-2 text-muted-foreground line-clamp-2">
              {member.alignment_reason}
            </p>
          )}
        </div>
      </div>
      
      {onActionClick && (
        <div className="px-4 pb-4 mt-auto">
          <Button 
            variant="outline" 
            size="sm" 
            className="w-full"
            onClick={() => {
              logAlignmentClick();
              onActionClick();
            }}
          >
            {actionText}
          </Button>
        </div>
      )}
    </Card>
  );
} 