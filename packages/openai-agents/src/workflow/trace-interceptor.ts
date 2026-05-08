import { getCurrentTrace, withCustomSpan, createCustomSpan } from '@openai/agents-core';
import type {
  ActivityInput,
  LocalActivityInput,
  Next,
  StartChildWorkflowExecutionInput,
  SignalWorkflowInput,
  WorkflowExecuteInput,
  SignalInput,
  QueryInput,
  UpdateInput,
  WorkflowInterceptors,
  WorkflowInboundCallsInterceptor,
  WorkflowOutboundCallsInterceptor,
} from '@temporalio/workflow';
import type { Headers } from '@temporalio/workflow';
import { currentAgentsSpanHeader, extractAgentsTraceHeader, injectAgentsTraceHeader } from '../common/trace-header';
import { withRestoredAgentsTraceContext, withRestoredAgentsTraceContextSync } from '../common/trace-context';

// Config symbol — set by ensureTracingProcessorRegistered() in tracing.ts
const CONFIG_SYMBOL = Symbol.for('temporal-openai-agents-config');

interface TracingConfig {
  addTemporalSpans?: boolean;
  startTraces?: boolean;
}

function getTracingConfig(): TracingConfig {
  return (globalThis as any)[CONFIG_SYMBOL] ?? {};
}

function shouldAddTemporalSpans(): boolean {
  return getTracingConfig().addTemporalSpans === true;
}

/**
 * Conditionally wraps `fn` in a `withCustomSpan` call gated on `addTemporalSpans`.
 */
async function maybeTemporalSpan<T>(
  spanName: string,
  fn: () => Promise<T>,
  data?: Record<string, unknown>
): Promise<T> {
  if (shouldAddTemporalSpans() && getCurrentTrace()) {
    return withCustomSpan(fn, { data: { name: spanName, data: data ?? {} } });
  }
  return fn();
}

/**
 * Guards on an active trace, optionally wraps in a temporal span, then injects
 * the current agent trace/span header into the outbound input's headers and
 * calls `next`.
 *
 * Returns `next(input)` unmodified when no agent trace is active.
 *
 * Header capture happens inside the optional temporal span callback so that when
 * `addTemporalSpans` is enabled, the propagated spanId is the temporal span
 * itself — the receiving side uses this spanId to derive its OTel parent.
 *
 * `currentAgentsSpanHeader()` is called without a null guard because it returns
 * null only when `getCurrentTrace()` is null, which the outer guard already
 * excludes.
 */
async function withInjectedHeader<I extends { headers: Headers }, T>(
  input: I,
  next: (i: I) => Promise<T>,
  spanConfig?: { name: string; data?: Record<string, unknown> }
): Promise<T> {
  if (!getCurrentTrace()) return next(input);

  const doInject = (): Promise<T> => {
    const header = currentAgentsSpanHeader()!;
    const headers = injectAgentsTraceHeader(input.headers, header);
    return next({ ...input, headers });
  };

  if (spanConfig) {
    return maybeTemporalSpan(spanConfig.name, doInject, spanConfig.data);
  }
  return doInject();
}

// --- Workflow Inbound Interceptor ---

/**
 * Workflow inbound interceptor that restores OpenAI Agents trace context
 * from propagated headers on workflow execute, signal, query, and update.
 *
 */
export class OpenAIAgentsTraceInboundInterceptor implements WorkflowInboundCallsInterceptor {
  async execute(input: WorkflowExecuteInput, next: Next<WorkflowInboundCallsInterceptor, 'execute'>): Promise<unknown> {
    const header = extractAgentsTraceHeader(input.headers);
    if (!header?.traceId) {
      return maybeTemporalSpan('temporal:executeWorkflow', () => next(input));
    }

    return withRestoredAgentsTraceContext(
      header,
      () => maybeTemporalSpan('temporal:executeWorkflow', () => next(input)),
      { startTraces: getTracingConfig().startTraces }
    );
  }

  async handleSignal(input: SignalInput, next: Next<WorkflowInboundCallsInterceptor, 'handleSignal'>): Promise<void> {
    const header = extractAgentsTraceHeader(input.headers);
    if (!header?.traceId) {
      return maybeTemporalSpan('temporal:handleSignal', () => next(input), {
        signalName: input.signalName,
      });
    }

    return withRestoredAgentsTraceContext(
      header,
      () =>
        maybeTemporalSpan('temporal:handleSignal', () => next(input), {
          signalName: input.signalName,
        }),
      { startTraces: getTracingConfig().startTraces }
    );
  }

  async handleQuery(input: QueryInput, next: Next<WorkflowInboundCallsInterceptor, 'handleQuery'>): Promise<unknown> {
    // Queries are read-only and don't carry propagated trace context.
    return maybeTemporalSpan('temporal:handleQuery', () => next(input), {
      queryName: input.queryName,
    });
  }

  validateUpdate(input: UpdateInput, next: Next<WorkflowInboundCallsInterceptor, 'validateUpdate'>): void {
    // Uses the sync ALS.run path — AsyncLocalStorage.run() is synchronous.
    const header = extractAgentsTraceHeader(input.headers);
    if (!header?.traceId) return next(input);
    return withRestoredAgentsTraceContextSync(header, () => next(input));
  }

  async handleUpdate(
    input: UpdateInput,
    next: Next<WorkflowInboundCallsInterceptor, 'handleUpdate'>
  ): Promise<unknown> {
    const header = extractAgentsTraceHeader(input.headers);
    if (!header?.traceId) return next(input);

    return withRestoredAgentsTraceContext(header, () => next(input), {
      startTraces: getTracingConfig().startTraces,
    });
  }
}

// --- Workflow Outbound Interceptor ---

/**
 * Workflow outbound interceptor that injects the active OpenAI Agents
 * trace/span context into outbound activity, child workflow, and signal
 * headers under the `__openai_span` key. This propagates the agent's
 * trace tree across the workflow→activity boundary (paired with
 * `OpenAIAgentsTraceActivityInboundInterceptor`) and the
 * workflow→child-workflow / workflow→signal boundaries (paired with
 * `OpenAIAgentsTraceInboundInterceptor`).
 *
 */
export class OpenAIAgentsTraceOutboundInterceptor implements WorkflowOutboundCallsInterceptor {
  async scheduleActivity(
    input: ActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleActivity'>
  ): Promise<unknown> {
    return withInjectedHeader(input, next, {
      name: `temporal:startActivity:${input.activityType}`,
      data: { activityType: input.activityType },
    });
  }

  async scheduleLocalActivity(
    input: LocalActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleLocalActivity'>
  ): Promise<unknown> {
    return withInjectedHeader(input, next, {
      name: `temporal:startLocalActivity:${input.activityType}`,
      data: { activityType: input.activityType },
    });
  }

  async startChildWorkflowExecution(
    input: StartChildWorkflowExecutionInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'startChildWorkflowExecution'>
  ): Promise<[Promise<string>, Promise<unknown>]> {
    if (!getCurrentTrace()) return next(input);

    // Header injection is shared between both branches — build the
    // modified input once.
    const header = currentAgentsSpanHeader()!;
    const headers = injectAgentsTraceHeader(input.headers, header);
    const injectedInput = { ...input, headers };

    if (shouldAddTemporalSpans()) {
      // Manual createCustomSpan + start/end because withCustomSpan doesn't
      // fit the [Promise<string>, Promise<unknown>] tuple return shape —
      // the span must live across both promises.
      const span = createCustomSpan({
        data: { name: `temporal:startChildWorkflow:${input.workflowType}`, data: { workflowType: input.workflowType } },
      });
      span.start();
      try {
        const [startedPromise, resultPromise] = await next(injectedInput);
        resultPromise.finally(() => span.end()).catch(() => {});
        return [startedPromise, resultPromise];
      } catch (e) {
        span.end();
        throw e;
      }
    }

    return next(injectedInput);
  }

  async signalWorkflow(
    input: SignalWorkflowInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'signalWorkflow'>
  ): Promise<void> {
    return withInjectedHeader(input, next, {
      name: 'temporal:signalWorkflow',
      data: { signalName: input.signalName },
    });
  }
}

export const interceptors = (): WorkflowInterceptors => ({
  inbound: [new OpenAIAgentsTraceInboundInterceptor()],
  outbound: [new OpenAIAgentsTraceOutboundInterceptor()],
});
