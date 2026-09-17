# GitHub Integration

## Overview

The Prothynesis Runtime can optionally integrate with GitHub for user attribution. This integration is used solely to identify contributors and attribute solver training work. No personal data beyond your public GitHub profile is collected or stored.

## Why GitHub

GitHub is used as the identity provider because:
- It is widely used by the machine learning community
- It provides a standard OAuth flow without requiring passwords
- It allows optional attribution without mandatory account creation

**You do not need a GitHub account to use the Prothynesis Runtime.** The runtime operates fully without authentication. GitHub integration is only needed if you want your training contributions to be attributed to your profile.

## OAuth Flow

The runtime uses GitHub's OAuth 2.0 web application flow:

1. The runtime opens a browser window to GitHub's authorization page
2. You authorize the application to read your public profile
3. GitHub redirects back to the runtime with an authorization code
4. The runtime exchanges the code for an access token
5. The token is stored encrypted locally
6. The runtime fetches your public profile information

The access token is only used to fetch your public GitHub profile. It is never used to access private repositories or make changes on your behalf.

## App Authorization

The runtime requests the following GitHub scope:
- `read:user` - Read your public profile information

No additional scopes are requested. The application cannot:
- Access your private repositories
- Create or modify issues, pull requests, or gists
- Access your organizations or teams
- Read your email address or other private data

## Credential Storage

GitHub credentials are stored locally and encrypted:
- The access token is encrypted using a Fernet symmetric key
- The encryption key is stored in a file with restricted permissions (0600)
- Credentials are never transmitted to third parties
- Credentials can be cleared at any time with `prothynesis auth logout`

## Attribution

When you contribute a solver checkpoint as a training worker:
- Your GitHub username is recorded in the solver metadata
- Your contribution appears in the community leaderboard
- You receive credit in model metadata and release notes

Attribution is opt-in. You can train solvers without linking a GitHub account.

## Disconnect

To disconnect your GitHub account:

```bash
prothynesis auth logout
```

This removes all stored credentials and stops attribution of future contributions. Past contributions will retain attribution unless manually removed by the coordinator.

## Privacy

The Prothynesis Runtime respects your privacy:

- **No PAT required**: Personal Access Tokens are not used or required
- **No password collection**: Your GitHub password is never collected or stored
- **Minimal data**: Only your public GitHub profile (username, ID, avatar) is accessed
- **Local storage**: All credentials are stored encrypted on your machine
- **Optional**: GitHub integration is completely optional
- **Transparent**: The OAuth flow uses standard GitHub endpoints and can be audited

If you have privacy concerns, you can use the runtime without any authentication.
