import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { User } from '@supabase/supabase-js'
import Link from 'next/link'

interface ProfileCompletionProps {
  user: User
}

export function ProfileCompletion({ user }: ProfileCompletionProps) {
  // Calculate profile completion percentage based on filled fields
  const profileFields = [
    { name: 'Avatar', completed: !!user.user_metadata?.avatar_url },
    { name: 'Full Name', completed: !!user.user_metadata?.full_name },
    { name: 'Bio', completed: !!user.user_metadata?.bio },
    { name: 'Interests', completed: !!user.user_metadata?.interests },
  ]

  const completedFields = profileFields.filter(field => field.completed).length
  const completionPercentage = (completedFields / profileFields.length) * 100

  return (
    <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Profile Completion</h3>
        <Button variant="outline" asChild>
          <Link href="/dashboard/settings">Complete Profile</Link>
        </Button>
      </div>
      
      <Progress value={completionPercentage} className="h-2 mb-4" />
      
      <p className="text-sm text-muted-foreground mb-4">
        {completionPercentage === 100
          ? "Your profile is complete! 🎉"
          : `Complete your profile to increase your chances of making meaningful connections (${completionPercentage}% complete)`}
      </p>

      <div className="space-y-2">
        {profileFields.map(field => (
          <div key={field.name} className="flex items-center">
            <div className={`w-2 h-2 rounded-full mr-2 ${field.completed ? 'bg-green-500' : 'bg-gray-300'}`} />
            <span className="text-sm">{field.name}</span>
          </div>
        ))}
      </div>
    </div>
  )
} 