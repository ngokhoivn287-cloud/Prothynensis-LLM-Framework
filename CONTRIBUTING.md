# Contributing

Thank you for your interest in contributing to Prothynesis. This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating, you agree to uphold the standards described in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

- Search existing issues before opening a new one.
- Include steps to reproduce, expected behavior, and actual behavior.
- Include environment details (OS, Python version, hardware).

### Suggesting Features

- Open an issue describing the feature and its motivation.
- Discuss design trade-offs before implementing large changes.

### Development Setup

```bash
# Core framework
cd modular-moe
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Runtime
cd ../prothynesis-runtime
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests

```bash
# Core framework
cd modular-moe
pytest tests/ -v

# Runtime
cd prothynesis-runtime
pytest tests/ -v
```

### Code Style

- Follow PEP 8.
- Keep functions focused and small.
- Add tests for new behavior.
- Update documentation for user-facing changes.

### Commit Messages

Use clear, present-tense messages. Reference issues when applicable.

## Review Process

- Maintainers will review pull requests.
- Address review comments promptly.
- Approved changes are merged by maintainers.
