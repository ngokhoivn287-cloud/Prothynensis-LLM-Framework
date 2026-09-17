# Security

## Principles

The Prothynesis Runtime is designed with security as a foundational requirement. The following principles guide all security decisions:

1. **Reject unsigned or malformed jobs**: All jobs must be properly signed and validated before execution
2. **Never execute arbitrary code**: Training runs are constrained to deterministic operations within the job specification
3. **Verify artifact integrity**: All artifacts are checksum-verified before use
4. **Isolate worker execution**: Each worker runs in a constrained environment with conservative resource limits
5. **Protect credentials**: Authentication tokens are encrypted at rest with restricted permissions

## Job Validation

Every job received by a worker is validated before execution. Validation includes:

- Schema validation: All required fields are present and correctly typed
- Signature verification: The job signature is verified against the coordinator's public key
- Runtime version check: The job runtime version is in the supported list
- Architecture check: The job target architecture matches the worker's capabilities
- Solver ID format: The solver ID matches the expected pattern

Jobs that fail any validation check are rejected immediately and reported to the coordinator.

## Signature Verification

Job signatures ensure authenticity and integrity. The verification process:

1. Extract the signature from the job payload
2. Recompute the canonical JSON representation of the job
3. Verify the signature against the coordinator's public key
4. Reject the job if verification fails

The coordinator's public key is distributed with the runtime and rotated periodically.

## Artifact Integrity

All artifacts (datasets, checkpoints, packages) are verified using SHA256 checksums:

- Download integrity: Files are checksummed after download
- Storage integrity: Stored artifacts are periodically re-verified
- Submission integrity: Packages include checksums verified by the coordinator
- Extraction integrity: Extracted files are validated against expected checksums

Corrupted or tampered artifacts are rejected and reported.

## Worker Isolation

Workers are isolated from each other and from the host system:

- Each worker has its own artifact store directory
- Training processes are monitored for resource limits
- Workers cannot access files outside their designated directories
- Path validation prevents directory traversal attacks
- Workers pause when host resources are constrained

## API Security

The local API server is bound to `127.0.0.1` by default and is not exposed to external networks. API security measures include:

- No authentication required for local access (trusted localhost)
- Input validation on all endpoints
- Rate limiting on inference endpoints
- No arbitrary code execution via API

If you need remote access, use an SSH tunnel rather than exposing the API to public interfaces.

## Credential Storage

GitHub OAuth credentials are stored with the following protections:

- Encrypted with Fernet symmetric encryption
- Encryption key stored with file permission 0600 (owner read/write only)
- Credentials are cleared on logout
- Expired tokens are automatically cleared
- Credentials are never logged or transmitted in plaintext

## Path Validation

All file path operations are validated to prevent directory traversal:

- Paths are resolved to absolute paths before validation
- Operations outside the designated artifact store are rejected
- Symlinks are not followed for security-sensitive paths
- File existence is verified before operations

This prevents malicious jobs from accessing or modifying files outside their intended scope.
