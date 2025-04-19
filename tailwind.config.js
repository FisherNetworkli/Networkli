/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: ['class'],
    content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
  	extend: {
  		fontFamily: {
  			sans: [
  				'var(--font-inter)'
  			],
  			museo: [
  				'Museo Sans',
  				'sans-serif'
  			],
  			raleway: [
  				'Raleway Light',
  				'sans-serif'
  			]
  		},
  		colors: {
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			'networkli-orange': {
  				'40': '#F7AD93',
  				'70': '#F4845D',
  				DEFAULT: '#F15B27'
  			},
  			'connection-blue': {
  				'40': '#A9B9DF',
  				'70': '#6F89C4',
  				DEFAULT: '#3659A8'
  			},
  			chart: {
  				'1': 'hsl(var(--chart-1))',
  				'2': 'hsl(var(--chart-2))',
  				'3': 'hsl(var(--chart-3))',
  				'4': 'hsl(var(--chart-4))',
  				'5': 'hsl(var(--chart-5))'
  			}
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		}
  	}
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
    require('@tailwindcss/aspect-ratio'),
    require("tailwindcss-animate"),
    // Custom components for cohesive app-like styling
    function({ addComponents, theme }) {
      addComponents({
        '.card-frosted': {
          'background-color': 'rgba(255, 255, 255, 0.2)',
          'backdrop-filter': 'blur(16px)',
          'border-radius': theme('borderRadius.lg'),
          'box-shadow': '0 10px 30px -10px rgba(0,0,0,0.1)',
          padding: theme('spacing.6'),
        },
        '.section': {
          padding: `${theme('spacing.6')} ${theme('spacing.4')}`,
          '@screen md': {
            padding: `${theme('spacing.8')} ${theme('spacing.8')}`,
          }
        },
        '.button-primary': {
          '@apply inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-[rgb(var(--connection-blue))] hover:bg-[rgb(var(--connection-blue-70))] transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[rgb(var(--connection-blue-70))]': {},
        }
      })
    }
  ],
} 