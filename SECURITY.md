# Security Policy

## Scope

Xurcroh is under pre-GPU engineering validation. It must not be described as production-certified or cryptographically proven secure. Basis transformations are computational obfuscation and do not replace confidential computing, cryptographic secret sharing, or an independent security review.

## Reporting

Report suspected vulnerabilities privately to the Project Manager and repository security contact. Include the affected revision, a minimal reproduction, impact, and whether credentials or private data are required for validation. Do not open a public issue containing exploit details, prompts, keys, provider credentials, or private model artifacts.

## Operational requirements

Use strict shard loading and authenticated transport, keep prefix caching disabled unless a reviewed isolated implementation is explicitly selected, pin worker identities and model revisions, and never place secrets in configuration files committed to Git. Hardware attestation is optional defense-in-depth and has documented physical and trust-root limitations.
