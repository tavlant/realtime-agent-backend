# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-30

### Added
- Alembic migrations (`migrations/`), applied before the API starts in Docker Compose, and a test
  that fails if the ORM models and the migrations drift apart.
- `DB_AUTO_CREATE` setting: `create_all` stays for dev and tests, deployments use migrations.
- `mypy --strict` in CI and pre-commit; `py.typed` marker.
- Coverage reporting in CI with an 85% minimum.
- `uv.lock` lockfile; CI and the Docker image install from it.
- Architecture docs and decision records (`docs/`), pre-commit hooks, Dependabot, PR template,
  `.editorconfig`, `.gitattributes`, `SECURITY.md`.

### Changed
- `ConversationRepository.list` renamed to `list_conversations` (it shadowed the built-in
  `list` inside the class, which hid type errors).
- README trimmed to an overview; details moved to `docs/`.

## [0.1.0] - 2026-10-30

### Added
- Streaming chat agents over WebSockets with tool calling, Redis + Postgres memory, rolling
  summaries, per-conversation locking, pgvector RAG, Anthropic/OpenAI/Fake providers, CI and Docker.

[0.2.0]: https://github.com/tavlant/realtime-agent-backend/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tavlant/realtime-agent-backend/releases/tag/v0.1.0
