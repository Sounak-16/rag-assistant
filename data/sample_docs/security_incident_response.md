# Security Incident Response Plan & Severity Guidelines

## Severity Classifications and Response Windows

### Severity 1 (Sev-1): Critical Security Incident
- Criteria: Active compromise of production environments, unauthorized exfiltration of sensitive customer data or PII, total catastrophic outage affecting all regions, or active ransomware attack.
- Initial Response Window: Within 15 minutes of trigger or detection (24/7/365).
- Incident Commander: Vice President of Infrastructure or Chief Information Security Officer (CISO).
- Mandatory Notifications: Immediate executive paging, General Counsel (Legal), and external customer notification within 72 hours per regulatory compliance.

### Severity 2 (Sev-2): Major Security Incident
- Criteria: Vulnerability actively exploited in a non-production staging environment, partial outage of customer-facing components, or security control bypass with isolated scope.
- Initial Response Window: Within 1 hour of alert generation.
- Incident Commander: On-call Security Lead.
- Mandatory Notifications: Director of Security and Engineering VP within 4 hours.

### Severity 3 (Sev-3): Minor Incident
- Criteria: Non-exploitable vulnerability detected by automated scanner, isolated phishing attempt blocked by email filters, or internal policy infraction.
- Initial Response Window: Within 24 hours during standard business days.

## Incident Escalation Workflow
All incident reports must be initiated through the PagerDuty security service or by typing `/incident declare` in the dedicated #sec-ops Slack channel.

## Post-Mortem Requirements
- Every Sev-1 and Sev-2 incident requires a blameless post-mortem retrospective meeting held within 3 business days of incident resolution.
- The root cause analysis (RCA) document must be signed off by the Incident Commander and published to the company internal engineering wiki within 5 business days.
