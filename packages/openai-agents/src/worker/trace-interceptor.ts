import { withCustomSpan } from '@openai/agents-core';
import type { Context as ActivityContext } from '@temporalio/activity';
import type { Next, ActivityInboundCallsInterceptor, ActivityExecuteInput } from '@temporalio/worker';
import { extractAgentsTraceHeader } from '../common/trace-header';
import { withRestoredAgentsTraceContext } from '../common/trace-context';

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

  /**
   * When `true` (default), wraps intercepted calls in `temporal:*` custom spans
   * for Temporal-specific instrumentation (e.g. `temporal:executeActivity`).
   * Set to `false` to disable these spans while keeping trace context propagation.
   *
   * Mirrors Python's `add_temporal_spans` parameter.
   */
  addTemporalSpans?: boolean;
}

/**
 * Activity inbound interceptor that restores OpenAI Agents trace context
 * from propagated headers and optionally wraps activity execution in a
 * `temporal:executeActivity` span. Mirrors Python's
 * `_ContextPropagationActivityInboundInterceptor`.
 */
export class OpenAIAgentsTraceActivityInboundInterceptor implements ActivityInboundCallsInterceptor {
  private readonly ctx: ActivityContext;

  constructor(
    ctx: ActivityContext,
    private readonly options?: OpenAIAgentsTraceInterceptorOptions
  ) {
    this.ctx = ctx;
  }

  async execute(input: ActivityExecuteInput, next: Next<ActivityInboundCallsInterceptor, 'execute'>): Promise<unknown> {
    const header = extractAgentsTraceHeader(input.headers);
    if (!header?.traceId) return next(input);

    const addSpans = this.options?.addTemporalSpans !== false;

    return withRestoredAgentsTraceContext(
      header,
      async () => {
        if (addSpans) {
          const info = this.ctx.info;
          return withCustomSpan(() => next(input), {
            data: {
              name: 'temporal:executeActivity',
              data: { activityId: info.activityId, activityType: info.activityType },
            },
          });
        }
        return next(input);
      },
      { startTraces: this.options?.startTraces }
    );
  }
}
