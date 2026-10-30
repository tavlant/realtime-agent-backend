# Operations handbook

## Deployments
Deployments go out through the CI pipeline after lint and tests pass. Production
releases happen on Tuesdays and Thursdays; hotfixes may ship any day with a reviewer.

## On-call
The on-call engineer acknowledges pages within 15 minutes. Escalate to the team lead
if an incident is not mitigated within one hour.

## Secrets
API keys are rotated every 90 days and are never committed to the repository.
