# Repository Guidelines

## Project Structure & Module Organization

This is the Temporal TypeScript SDK pnpm workspace. Package source lives under `packages/<name>/src`, with build output in `lib`; do not edit generated `lib` files directly. Core public packages include `client`, `worker`, `workflow`, `activity`, `common`, `proto`, and `testing`. Cross-package integration tests live in `packages/test/src/test-*.ts`; package-local tests commonly live under `src/__tests__/test-*.ts`. Rust bridge code is in `packages/core-bridge/src`, shared scripts are in `scripts`, and optional integrations live under `contrib`.

## Build, Test, and Development Commands

Use Node 20, 22, or 24, Rust, and `protoc`. Prefer Corepack-managed pnpm:

- `corepack enable` - makes the pinned pnpm version available.
- `corepack pnpm install --frozen-lockfile` - installs workspace dependencies.
- `corepack pnpm build` - builds protos, native bridge pieces, and TypeScript packages.
- `corepack pnpm run rebuild` - cleans generated output and rebuilds from scratch.
- `corepack pnpm lint:check` - checks ESLint and Prettier without rewriting files.
- `corepack pnpm -F @temporalio/worker... run build` - builds one package plus dependencies.
- `corepack pnpm -F @temporalio/common run test` - runs a targeted package test suite.

## Coding Style & Naming Conventions

TypeScript is formatted with Prettier and linted with ESLint. Keep imports ordered by the configured groups, use type-only imports when appropriate, and avoid importing package internals through `@temporalio/*/src/*`. Public APIs should have explicit module boundary types. Prefer clear names and straightforward control flow over dense helper layers.

## Testing Guidelines

Tests use AVA. Add package-local unit tests as `src/__tests__/test-*.ts` when the behavior belongs to one package, and add cross-SDK scenarios as `packages/test/src/test-*.ts`. Run the narrowest useful command first, then broaden to `corepack pnpm test` when changes affect shared behavior. Integration tests require a local Temporal server; set `RUN_INTEGRATION_TESTS=true` only when running those scenarios.

## Commit & Pull Request Guidelines

Use Conventional Commit-style titles, for example `fix(worker): handle replay shutdown`. Valid scopes are enforced by `commitlint.config.js` and include `activity`, `client`, `core`, `docs`, `proto`, `worker`, and `workflow`. Start larger changes with a GitHub issue, sign the CLA, link the issue in the PR, describe the user-visible behavior, and list the commands you ran.

## Security & Configuration Tips

Do not commit credentials, local Temporal data, generated runtime state, or machine-specific paths. Keep dependency and submodule updates intentional and mention them explicitly in the PR.
