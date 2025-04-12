# Read-Only Mode for Networkli

This document explains how to use the read-only mode feature in Networkli.

## What is Read-Only Mode?

Read-only mode is a feature that prevents changes to the public (logged-out) portion of the site while still allowing authenticated users to make changes. This is useful when you want to:

- Prevent unauthorized changes to the public-facing content
- Allow development to continue on the authenticated portion of the site
- Maintain the site's appearance while making backend changes

## How to Enable/Disable Read-Only Mode

### Using the Toggle Script

The easiest way to toggle read-only mode is to use the provided script:

```bash
./scripts/toggle-readonly.sh
```

This script will:
1. Toggle the `READONLY_MODE` environment variable in `.env.local`
2. Restart the development server

### Manual Configuration

You can also manually configure read-only mode by editing the environment files:

1. For local development, edit `.env.local`:
   ```
   READONLY_MODE=true
   ```

2. For production, edit `.env.production`:
   ```
   READONLY_MODE=true
   ```

## How Read-Only Mode Works

Read-only mode works by:

1. Blocking write operations (POST, PUT, DELETE) to public API routes
2. Allowing all operations for authenticated users
3. Displaying a banner to inform users that the site is in read-only mode

## What is Blocked in Read-Only Mode?

In read-only mode, the following operations are blocked for unauthenticated users:

- Any POST, PUT, DELETE requests to public API routes
- Form submissions on public pages
- Any other write operations to the public portion of the site

## What is Allowed in Read-Only Mode?

In read-only mode, the following operations are still allowed:

- All read operations (GET requests)
- All operations for authenticated users
- Authentication-related operations (login, signup, etc.)

## Troubleshooting

If you encounter issues with read-only mode:

1. Check that the `READONLY_MODE` environment variable is set correctly
2. Restart the development server
3. Clear your browser cache
4. Check the browser console for any errors

## Need Help?

If you need help with read-only mode, please contact the development team. 