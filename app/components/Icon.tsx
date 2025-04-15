import { cn } from '@/lib/utils'

export type IconName = 
  | 'progress-circle'
  | 'progress-circle-empty'
  | 'arrow-right'
  | 'arrow-left'
  | 'check'
  | 'error'
  | 'loading'

interface IconProps extends React.SVGProps<SVGSVGElement> {
  name: IconName
  size?: number
  className?: string
}

export function Icon({ name, size = 24, className, ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      className={cn('inline-block', {
        'animate-spin': name === 'loading'
      }, className)}
      {...props}
    >
      <use href={`/images/icons/form-icons.svg#${name}`} />
    </svg>
  )
} 