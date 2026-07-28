# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Tributo, **please do not file a
public issue**. Instead, report it privately via GitHub's Security Advisory
system:

1. Go to the Security tab of the repository.
2. Click **Report a vulnerability**.
3. Provide a detailed description, including steps to reproduce.

We will acknowledge your report within 5 business days and aim to provide a fix
or mitigation within 30 days.

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |

## Security Best Practices for Users

- Use environment variables for credentials — never hard-code secrets in
  training configs.
- Run Ray clusters with firewall rules restricting dashboard access.
- Keep Tributo and its dependencies up to date.
