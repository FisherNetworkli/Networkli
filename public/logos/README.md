# Networkli Logo Files

Place the following logo files in this directory:

1. `networkli-default.svg` - Primary logo in Connection Blue (#3659A8)
2. `networkli-black.svg` - Black version of the logo
3. `networkli-orange.svg` - Orange version of the logo (#F15B27)
4. `networkli-white.svg` - White version of the logo
5. `networkli-icon.svg` - App icon/square logo

## Logo Guidelines

- Use SVG format for all logos to ensure scalability
- Maintain the original proportions
- Follow the spacing and clear space guidelines from the brand guide
- Use the correct color values:
  - Networkli Orange: #F15B27
  - Connection Blue: #3659A8

## Usage in Code

The Logo component (`components/Logo.tsx`) will automatically handle the different logo variants. Use it like this:

```tsx
import Logo from '@/components/Logo'

// Default blue logo
<Logo />

// Other variants
<Logo variant="black" />
<Logo variant="orange" />
<Logo variant="white" />
<Logo variant="icon" />

// With custom classes
<Logo variant="default" className="w-48" />
``` 