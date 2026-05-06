import {
  Trace,
  getCurrentTrace,
  getGlobalTraceProvider,
  withTrace,
  setCurrentSpan,
  type CustomSpanData,
} from '@openai/agents-core';
import type { Context as ActivityContext } from '@temporalio/activity';
import type { Next, ActivityInboundCallsInterceptor, ActivityExecuteInput } from '@temporalio/worker';
import { extractAgentsTraceHeader } from '../common/trace-header';

// Stable cross-package contract symbol from @openai/agents-core.
// The library stores a single AsyncLocalStorage instance on globalThis under this
// symbol to share trace context across duplicate package installations. If upstream
// changes this symbol, this interceptor will silently stop propagating agent trace
// context — but the symbol has been stable since the library's initial release and
// is documented in source comments as an intentional dedup mechanism.
const AGENTS_CORE_ALS_SYMBOL = Symbol.for('openai.agents.core.asyncLocalStorage');

export interface OpenAIAgentsTraceInterceptorOptions {
  /**
   * When `true`, calls `trace.start()` / `span.start()` on the restored context,
   * which fires `onTraceStart` / `onSpanStart` on registered `TracingProcessor`s.
   * When `false` (default), sets the AsyncLocalStorage context directly without
   * processor events — sufficient for ID propagation without duplicate trace entries.
   *
   * Set to `true` if you register activity-side `TracingProcessor`s that need to
   * see trace lifecycle events. The default `false` avoids duplication when only the
   * workflow-side `TemporalTracingProcessor` is active.
   *
   * Default: `false` (matches Python's `start_traces=False`).
   */
  startTraces?: boolean;
}

export class OpenAIAgentsTraceActivityInboundInterceptor implements ActivityInboundCallsInterceptor {
  constructor(
    _ctx: ActivityContext,
    private readonly options?: OpenAIAgentsTraceInterceptorOptions
  ) {}

  async execute(input: ActivityExecuteInput, next: Next<ActivityInboundCallsInterceptor, 'execute'>): Promise<unknown> {
    const header = extractAgentsTraceHeader(input.headers);
    if (!header?.traceId) return next(input);

    if (this.options?.startTraces) {
      return this.executeWithTraceEvents(header.traceId, header.traceName, header.spanId, input, next);
    }
    return this.executeWithContextOnly(header.traceId, header.traceName, header.spanId, input, next);
  }

  private async executeWithTraceEvents(
    traceId: string,
    traceName: string,
    spanId: string | null,
    input: ActivityExecuteInput,
    next: Next<ActivityInboundCallsInterceptor, 'execute'>
  ): Promise<unknown> {
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
          return await next(input);
        } finally {
          span.end();
        }
      }
      return next(input);
    });
  }

  private async executeWithContextOnly(
    traceId: string,
    traceName: string,
    spanId: string | null,
    input: ActivityExecuteInput,
    next: Next<ActivityInboundCallsInterceptor, 'execute'>
  ): Promise<unknown> {
    // Force ALS initialization — @openai/agents-core lazily creates the
    // AsyncLocalStorage instance on first call to getContextAsyncLocalStorage().
    getCurrentTrace();

    const als = (globalThis as any)[AGENTS_CORE_ALS_SYMBOL] as
      | { run: <R>(store: unknown, callback: () => R) => R }
      | undefined;
    if (!als) return next(input);

    const trace = new Trace({ traceId, name: traceName });
    let span: unknown;
    if (spanId) {
      span = getGlobalTraceProvider().createSpan<CustomSpanData>(
        { spanId, data: { type: 'custom', name: '', data: {} } },
        trace
      );
    }

    return als.run({ trace, span, active: true }, () => next(input));
  }
}
