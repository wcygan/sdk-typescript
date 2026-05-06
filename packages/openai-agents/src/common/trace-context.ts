/**
 * Shared helper for restoring OpenAI Agents trace context from propagated headers.
 *
 * Used by both the workflow-inbound interceptor (V8 sandbox) and the
 * activity-inbound interceptor (worker host). Both runtimes bundle
 * `@openai/agents-core`, so the ALS and Trace/Span APIs are available
 * in either context.
 */
import {
  Trace,
  getCurrentTrace,
  getGlobalTraceProvider,
  setTracingDisabled,
  withTrace,
  setCurrentSpan,
  type CustomSpanData,
} from '@openai/agents-core';
import type { AgentsSpanHeader } from './trace-header';

// Stable cross-package contract symbol from @openai/agents-core.
// The library stores a single AsyncLocalStorage instance on globalThis under
// this symbol to share trace context across duplicate package installations.
const AGENTS_CORE_ALS_SYMBOL = Symbol.for('openai.agents.core.asyncLocalStorage');

export interface RestoreTraceContextOptions {
  /**
   * When `true`, calls `trace.start()` / `span.start()` on the restored
   * context, firing `onTraceStart`/`onSpanStart` on registered processors.
   * When `false` (default), sets the AsyncLocalStorage context directly
   * without processor events — sufficient for ID propagation.
   */
  startTraces?: boolean;
}

/**
 * Run `fn` within a restored OpenAI Agents trace context extracted from
 * a propagated header. If `header.traceId` is null, `fn` runs directly.
 */
export async function withRestoredAgentsTraceContext<T>(
  header: AgentsSpanHeader,
  fn: () => Promise<T>,
  options?: RestoreTraceContextOptions
): Promise<T> {
  if (!header.traceId) return fn();

  if (options?.startTraces) {
    return withTraceEvents(header.traceId, header.traceName, header.spanId, fn);
  }
  return withContextOnly(header.traceId, header.traceName, header.spanId, fn);
}

async function withTraceEvents<T>(
  traceId: string,
  traceName: string,
  spanId: string | null,
  fn: () => Promise<T>
): Promise<T> {
  // @openai/agents-core disables tracing in three cases: NODE_ENV=test,
  // browser environments, and OPENAI_AGENTS_DISABLE_TRACING=true.
  // startTraces=true is an explicit user opt-in to processor events,
  // so we override the library default before creating the trace.
  setTracingDisabled(false);
  const trace = getGlobalTraceProvider().createTrace({ traceId, name: traceName });
  return withTrace(trace, async () => {
    if (spanId) {
      const span = getGlobalTraceProvider().createSpan<CustomSpanData>({
        spanId,
        data: { type: 'custom', name: '', data: {} },
      });
      span.start();
      setCurrentSpan(span);
      try {
        return await fn();
      } finally {
        span.end();
      }
    }
    return fn();
  });
}

/**
 * Synchronous variant of {@link withRestoredAgentsTraceContext} for use in
 * synchronous interceptor hooks (e.g. `validateUpdate`). Only supports the
 * ALS-direct path (`startTraces` is always `false`).
 *
 * `AsyncLocalStorage.run()` is itself synchronous — it propagates the store
 * to both sync and async code within the callback and returns whatever the
 * callback returns.
 */
export function withRestoredAgentsTraceContextSync<T>(header: AgentsSpanHeader, fn: () => T): T {
  if (!header.traceId) return fn();
  return withContextOnlySync(header.traceId, header.traceName, header.spanId, fn);
}

function withContextOnlySync<T>(traceId: string, traceName: string, spanId: string | null, fn: () => T): T {
  getCurrentTrace();

  const als = (globalThis as any)[AGENTS_CORE_ALS_SYMBOL] as
    | { run: <R>(store: unknown, callback: () => R) => R }
    | undefined;
  if (!als) return fn();

  const trace = new Trace({ traceId, name: traceName });
  let span: unknown;
  if (spanId) {
    span = getGlobalTraceProvider().createSpan<CustomSpanData>(
      { spanId, data: { type: 'custom', name: '', data: {} } },
      trace
    );
  }

  return als.run({ trace, span, active: true }, fn);
}

async function withContextOnly<T>(
  traceId: string,
  traceName: string,
  spanId: string | null,
  fn: () => Promise<T>
): Promise<T> {
  // Force ALS initialization — @openai/agents-core lazily creates the
  // AsyncLocalStorage instance on first call to getContextAsyncLocalStorage().
  // Calling getCurrentTrace() is a side-effect-free read that triggers this.
  // See: @openai/agents-core/dist/tracing/context.js:getContextAsyncLocalStorage
  getCurrentTrace();

  const als = (globalThis as any)[AGENTS_CORE_ALS_SYMBOL] as
    | { run: <R>(store: unknown, callback: () => R) => R }
    | undefined;
  if (!als) return fn();

  const trace = new Trace({ traceId, name: traceName });
  let span: unknown;
  if (spanId) {
    span = getGlobalTraceProvider().createSpan<CustomSpanData>(
      { spanId, data: { type: 'custom', name: '', data: {} } },
      trace
    );
  }

  return als.run({ trace, span, active: true }, fn);
}
