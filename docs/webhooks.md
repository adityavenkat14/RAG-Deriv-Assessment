# Webhooks

Webhook delivery logs are retained for 30 days and can be viewed in the dashboard.
Each delivery log includes the event identifier, attempt timestamp, HTTP status, and response duration.
Failed webhook deliveries are retried up to 5 times over 24 hours.
Endpoints must return an HTTP 200 response within 10 seconds.
Verify webhook signatures with the signing secret shown in the dashboard.
