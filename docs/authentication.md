# Authentication

The ForgeNest API authentication methods support API keys and OAuth 2.0.
Use API keys for server integrations and OAuth 2.0 for delegated user access.
Send an API key in the Authorization header as a Bearer token.
OAuth access tokens expire after 60 minutes; refresh tokens expire after 30 days.
Keep credentials secret and rotate a compromised API key immediately.
