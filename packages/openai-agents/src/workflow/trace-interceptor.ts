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
    if (!getCurrentTrace()) return next(input);

    // Header capture is inside the temporal span callback so that the
    // propagated spanId is the `temporal:startActivity:*` span itself
    // (when addTemporalSpans is enabled). The activity side uses this
    // spanId to derive its OTel parent, nesting `temporal:executeActivity`
    // under `temporal:startActivity:*` in the trace tree.
    return maybeTemporalSpan(
      `temporal:startActivity:${input.activityType}`,
      () => {
        const header = currentAgentsSpanHeader();
        if (!header) return next(input);
        const headers = injectAgentsTraceHeader(input.headers, header);
        return next({ ...input, headers });
      },
      { activityType: input.activityType }
    );
  }

  async scheduleLocalActivity(
    input: LocalActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleLocalActivity'>
  ): Promise<unknown> {
    if (!getCurrentTrace()) return next(input);

    return maybeTemporalSpan(
      `temporal:startLocalActivity:${input.activityType}`,
      () => {
        const header = currentAgentsSpanHeader();
        if (!header) return next(input);
        const headers = injectAgentsTraceHeader(input.headers, header);
        return next({ ...input, headers });
      },
      { activityType: input.activityType }
    );
  }

  async startChildWorkflowExecution(
    input: StartChildWorkflowExecutionInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'startChildWorkflowExecution'>
  ): Promise<[Promise<string>, Promise<unknown>]> {
    if (!getCurrentTrace()) return next(input);

    if (shouldAddTemporalSpans()) {
      const span = createCustomSpan({
        data: { name: `temporal:startChildWorkflow:${input.workflowType}`, data: { workflowType: input.workflowType } },
      });
      span.start();
      try {
        const header = currentAgentsSpanHeader();
        const headers = header ? injectAgentsTraceHeader(input.headers, header) : input.headers;
        const [startedPromise, resultPromise] = await next({ ...input, headers });
        resultPromise.finally(() => span.end()).catch(() => {});
        return [startedPromise, resultPromise];
      } catch (e) {
        span.end();
        throw e;
      }
    }

    const header = currentAgentsSpanHeader();
    if (!header) return next(input);
    const headers = injectAgentsTraceHeader(input.headers, header);
    return next({ ...input, headers });
  }

  async signalWorkflow(
    input: SignalWorkflowInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'signalWorkflow'>
  ): Promise<void> {
    const header = currentAgentsSpanHeader();
    if (!header) return next(input);

    const headers = injectAgentsTraceHeader(input.headers, header);
    return maybeTemporalSpan('temporal:signalWorkflow', () => next({ ...input, headers }), {
      signalName: input.signalName,
    });
  }
}

export const interceptors = (): WorkflowInterceptors => ({
  inbound: [new OpenAIAgentsTraceInboundInterceptor()],
  outbound: [new OpenAIAgentsTraceOutboundInterceptor()],
});
