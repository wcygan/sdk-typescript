import { getCurrentTrace, withCustomSpan, createCustomSpan } from '@openai/agents-core';
import type {
  ActivityInput,
  ContinueAsNewInput,
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
import { workflowInfo } from '@temporalio/workflow';
import { currentAgentsSpanHeader, extractAgentsTraceHeader, injectAgentsTraceHeader } from '../common/trace-header';
import { extractAgentsConfigHeader, injectAgentsConfigHeader, CONFIG_WIRE_VERSION } from '../common/config-header';
import { withRestoredAgentsTraceContext, withRestoredAgentsTraceContextSync } from '../common/trace-context';
import { getCurrentPluginConfig, setPluginConfig, setOriginalPluginConfig, clearPluginConfig, type PluginConfig } from './plugin-config-store';
import { DEFAULT_MODEL_ACTIVITY_OPTIONS } from '../common/model-activity-options';

function shouldAddTemporalSpans(): boolean {
  return getCurrentPluginConfig()?.addTemporalSpans === true;
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
 * Workflow inbound interceptor that:
 * 1. Extracts the `__openai_agents_config` header on `execute` and populates
 *    the per-workflow plugin-config-store.
 * 2. Restores OpenAI Agents trace context from propagated headers on workflow
 *    execute, signal, query, and update.
 * 3. Clears the per-workflow config on workflow completion (try/finally).
 *
 * The config header is only injected by the client interceptor on workflow
 * start operations (`startWithDetails`, `signalWithStart`,
 * `startUpdateWithStart`); signal/query/update on existing workflows reuse
 * the config from the executing workflow's store entry.
 */
export class OpenAIAgentsTraceInboundInterceptor implements WorkflowInboundCallsInterceptor {
  /**
   * Extracts the config header, populates the per-workflow store, and restores
   * trace context from the propagated header.
   *
   * **Timing note on `startTraces`:** The initial `withRestoredAgentsTraceContext`
   * call reads `startTraces` from the just-extracted header config. If the
   * runner constructor later overrides `startTraces`, that override only takes
   * effect for subsequent inbound operations (signal, query, update) — NOT
   * the initial workflow execution entry. This is acceptable because the runner
   * hasn't been constructed yet at this point.
   */
  async execute(input: WorkflowExecuteInput, next: Next<WorkflowInboundCallsInterceptor, 'execute'>): Promise<unknown> {
    const wfId = workflowInfo().workflowId;

    // Extract config header and populate the per-workflow store.
    // The runner constructor may later override individual fields.
    const configHeader = extractAgentsConfigHeader(input.headers);
    if (configHeader) {
      const headerConfig: PluginConfig = {
        addTemporalSpans: configHeader.addTemporalSpans ?? false,
        startTraces: configHeader.startTraces ?? false,
        modelParams: { ...DEFAULT_MODEL_ACTIVITY_OPTIONS, ...configHeader.modelParams },
      };
      // Write to both maps: effective (for trace interceptor reads) and
      // original (for runner constructor merge base — immune to runner overwrites).
      setPluginConfig(wfId, headerConfig);
      setOriginalPluginConfig(wfId, headerConfig);
    }

    try {
      const header = extractAgentsTraceHeader(input.headers);
      if (!header?.traceId) {
        return await maybeTemporalSpan('temporal:executeWorkflow', () => next(input));
      }

      return await withRestoredAgentsTraceContext(
        header,
        () => maybeTemporalSpan('temporal:executeWorkflow', () => next(input)),
        { startTraces: getCurrentPluginConfig()?.startTraces }
      );
    } finally {
      clearPluginConfig(wfId);
    }
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
      { startTraces: getCurrentPluginConfig()?.startTraces }
    );
  }

  async handleQuery(input: QueryInput, next: Next<WorkflowInboundCallsInterceptor, 'handleQuery'>): Promise<unknown> {
    const header = extractAgentsTraceHeader(input.headers);
    if (!header?.traceId) {
      return maybeTemporalSpan('temporal:handleQuery', () => next(input), {
        queryName: input.queryName,
      });
    }

    return withRestoredAgentsTraceContext(
      header,
      () =>
        maybeTemporalSpan('temporal:handleQuery', () => next(input), {
          queryName: input.queryName,
        }),
      { startTraces: getCurrentPluginConfig()?.startTraces }
    );
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
      startTraces: getCurrentPluginConfig()?.startTraces,
    });
  }
}

// --- Workflow Outbound Interceptor ---

/**
 * Workflow outbound interceptor that injects the active OpenAI Agents
 * trace/span context into outbound activity, child workflow, signal, and
 * continueAsNew headers under the `__openai_span` key. For child workflows
 * and continueAsNew, also re-injects the `__openai_agents_config` header
 * so the new execution inherits the plugin config.
 *
 * The propagated config is the *effective* config (post-runner-overrides),
 * not the original header config. If a runner was constructed with overrides
 * (e.g. `addTemporalSpans: false`), those overrides flow into child
 * workflows and continueAsNew executions.
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
    // Re-inject config header for child workflows
    let headers = injectConfigHeaderFromStore(input.headers);

    if (!getCurrentTrace()) return next({ ...input, headers });

    // Header injection is shared between both branches — build the
    // modified input once.
    const header = currentAgentsSpanHeader()!;
    headers = injectAgentsTraceHeader(headers, header);
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

  async continueAsNew(
    input: ContinueAsNewInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'continueAsNew'>
  ): Promise<never> {
    // Re-inject both the config header and the trace header so the new
    // execution inherits the plugin config and trace context.
    let headers = injectConfigHeaderFromStore(input.headers);

    if (getCurrentTrace()) {
      const header = currentAgentsSpanHeader()!;
      headers = injectAgentsTraceHeader(headers, header);
    }

    return next({ ...input, headers });
  }
}

/**
 * Builds and injects a config header from the current workflow's plugin-config-store.
 * Used by outbound interceptor methods that start new workflow executions
 * (child workflows, continueAsNew).
 *
 * The runner-side `modelParams` accepts the full `ModelActivityOptions`, so a
 * `summaryOverride: ModelSummaryProvider` (function form) is a valid in-workflow
 * value. This function intentionally strips it before writing the wire header —
 * this is **wire-shape sanitization** (functions can't survive JSON serialization),
 * not a rejection of the runner's usage. The function form is consumed locally by
 * the runner's `ActivityBackedModel`; only string values propagate to children.
 */
function injectConfigHeaderFromStore(headers: Headers): Headers {
  const config = getCurrentPluginConfig();
  if (!config) return headers;

  // Narrow modelParams to SerializableModelActivityOptions for the wire header.
  // The runner may have merged in a ModelSummaryProvider (function form) for
  // summaryOverride — strip it here since functions can't survive JSON serialization.
  const { summaryOverride, ...restModelParams } = config.modelParams;
  const serializableModelParams = {
    ...restModelParams,
    ...(typeof summaryOverride === 'string' ? { summaryOverride } : {}),
  };

  return injectAgentsConfigHeader(headers, {
    __configVersion: CONFIG_WIRE_VERSION,
    addTemporalSpans: config.addTemporalSpans,
    startTraces: config.startTraces,
    modelParams: serializableModelParams,
  });
}

export const interceptors = (): WorkflowInterceptors => ({
  inbound: [new OpenAIAgentsTraceInboundInterceptor()],
  outbound: [new OpenAIAgentsTraceOutboundInterceptor()],
});
