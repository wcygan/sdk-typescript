import { withCustomSpan, getCurrentTrace } from '@openai/agents-core';
import type { Context as ActivityContext } from '@temporalio/activity';
import type { Next, ActivityInboundCallsInterceptor, ActivityExecuteInput } from '@temporalio/worker';
import { extractAgentsTraceHeader } from '../common/trace-header';
import { withRestoredAgentsTraceContext } from '../common/trace-context';
import { ensureActivityTracingProcessorRegistered } from './activity-tracing';

export interface OpenAIAgentsTraceInterceptorOptions {
  /**
   * When `true`, calls `trace.start()` / `span.start()` on the restored context,
   * which fires `onTraceStart` / `onSpanStart` on registered `TracingProcessor`s.
   * When `false` (default), sets the AsyncLocalStorage context directly without
   * processor events — sufficient for ID propagation without duplicate trace entries.
   *
   * Default: `false`.
   */
  startTraces?: boolean;

  /**
   * When `true`, wraps intercepted calls in `temporal:*` custom spans
   * for Temporal-specific instrumentation (e.g. `temporal:executeActivity`).
   * Set to `false` (default) to disable these spans while keeping trace
   * context propagation.
   *
   * The plugin propagates this value to workflows automatically via the
   * `__openai_agents_config` header. The workflow-side interceptor reads
   * its config from the per-workflow plugin-config-store (populated by the
   * inbound interceptor on header extraction). The runner constructor can
   * override this per-workflow if needed.
   *
   * Default: `false`.
   */
  addTemporalSpans?: boolean;
}

/**
 * Activity inbound interceptor that restores OpenAI Agents trace context
 * from propagated headers and optionally wraps activity execution in a
 * `temporal:executeActivity` agent SDK custom span. When `addTemporalSpans`
 * is enabled, the plugin registers an `ActivityTracingProcessor` that
 * bridges these spans to OTel.
 */
export class OpenAIAgentsTraceActivityInboundInterceptor implements ActivityInboundCallsInterceptor {
  private readonly ctx: ActivityContext;

  constructor(
    ctx: ActivityContext,
    private readonly options?: OpenAIAgentsTraceInterceptorOptions
  ) {
    this.ctx = ctx;

    // Register the activity-side OTel bridge so that withCustomSpan calls
    // in execute() produce OTel spans. Idempotent — only registers once
    // per worker process regardless of how many activity interceptors are created.
    if (this.options?.addTemporalSpans === true) {
      ensureActivityTracingProcessorRegistered();
    }
  }

  async execute(input: ActivityExecuteInput, next: Next<ActivityInboundCallsInterceptor, 'execute'>): Promise<unknown> {
    const header = extractAgentsTraceHeader(input.headers);
    if (!header?.traceId) return next(input);

    const addSpans = this.options?.addTemporalSpans === true;

    return withRestoredAgentsTraceContext(
      header,
      async () => {
        if (addSpans && getCurrentTrace()) {
          const info = this.ctx.info;
          return withCustomSpan(() => next(input), {
            data: {
              name: 'temporal:executeActivity',
              data: {
                activityId: info.activityId,
                activityType: info.activityType,
              },
            },
          });
        }
        return next(input);
      },
      { startTraces: this.options?.startTraces }
    );
  }
}
