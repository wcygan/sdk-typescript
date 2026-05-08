/**
 * Client-outbound interceptor for propagating OpenAI Agents trace context
 * and plugin config from the client to Temporal workflows, signals, queries,
 * and updates.
 */
import { getCurrentTrace, withCustomSpan } from '@openai/agents-core';
import type {
  WorkflowStartInput,
  WorkflowStartOutput,
  WorkflowSignalInput,
  WorkflowSignalWithStartInput,
  WorkflowQueryInput,
  WorkflowStartUpdateInput,
  WorkflowStartUpdateOutput,
  WorkflowStartUpdateWithStartInput,
  WorkflowStartUpdateWithStartOutput,
} from '@temporalio/client';
import { currentAgentsSpanHeader, injectAgentsTraceHeader } from '../common/trace-header';
import {
  CONFIG_WIRE_VERSION,
  injectAgentsConfigHeader,
  type AgentsConfigHeader,
} from '../common/config-header';
import type { SerializableModelActivityOptions } from '../common/model-activity-options';

export interface OpenAIAgentsTraceClientInterceptorOptions {
  /**
   * When `true`, wraps client calls (start workflow, signal,
   * query, update, signal-with-start, start-update-with-start) in
   * `temporal:*` custom spans.
   *
   * @default false
   */
  addTemporalSpans?: boolean;

  /**
   * When `true`, restored trace contexts fire processor events (`onTraceStart`,
   * `onSpanStart`). Propagated to the workflow via the config header.
   *
   * @default false
   */
  startTraces?: boolean;

  /**
   * Model activity options (timeouts, retry, task queue, etc.) propagated
   * from the plugin to the workflow via the config header. The runner's
   * `modelParams` override these per-field.
   *
   * Typed as {@link SerializableModelActivityOptions} — the function form of
   * `summaryOverride` (`ModelSummaryProvider`) is excluded at compile time
   * because it cannot survive JSON serialization through the config header.
   * Pass function-form overrides via the runner constructor in workflow code.
   */
  modelParams?: SerializableModelActivityOptions;
}

// Concrete `next` function types — avoids circular `Next<this, ...>` resolution
// that occurs when a class method's own signature feeds back into `Next`.
type NextStartWithDetails = (input: WorkflowStartInput) => Promise<WorkflowStartOutput>;
type NextSignal = (input: WorkflowSignalInput) => Promise<void>;
type NextQuery = (input: WorkflowQueryInput) => Promise<unknown>;
type NextStartUpdate = (input: WorkflowStartUpdateInput) => Promise<WorkflowStartUpdateOutput>;
type NextSignalWithStart = (input: WorkflowSignalWithStartInput) => Promise<string>;
type NextStartUpdateWithStart = (
  input: WorkflowStartUpdateWithStartInput
) => Promise<WorkflowStartUpdateWithStartOutput>;

// Structural typing — satisfies WorkflowClientInterceptor at assignment sites.
// Explicit `implements` is omitted because the recursive `Next<this, ...>`
// type in the interface causes circular resolution when used in class methods.
export class OpenAIAgentsTraceClientInterceptor {
  private readonly addTemporalSpans: boolean;
  private readonly configHeader: AgentsConfigHeader;

  /**
   * Captures a snapshot of the plugin config at construction time. The
   * resulting `configHeader` is injected into every workflow-starting call
   * (start, signalWithStart, startUpdateWithStart). Changes to plugin
   * options after construction are not reflected.
   */
  constructor(options?: OpenAIAgentsTraceClientInterceptorOptions) {
    this.addTemporalSpans = options?.addTemporalSpans === true;
    this.configHeader = {
      __configVersion: CONFIG_WIRE_VERSION,
      addTemporalSpans: options?.addTemporalSpans,
      startTraces: options?.startTraces,
      modelParams: options?.modelParams,
    };
  }

  private maybeSpan<T>(spanName: string, fn: () => Promise<T>, data?: Record<string, unknown>): Promise<T> {
    if (this.addTemporalSpans && getCurrentTrace()) {
      return withCustomSpan(() => fn(), { data: { name: spanName, data: data ?? {} } });
    }
    return fn();
  }

  async startWithDetails(input: WorkflowStartInput, next: NextStartWithDetails): Promise<WorkflowStartOutput> {
    // Always inject config header on workflow start (even without an active trace)
    let headers = injectAgentsConfigHeader(input.headers, this.configHeader);

    const header = currentAgentsSpanHeader();
    if (!header) return next({ ...input, headers });

    headers = injectAgentsTraceHeader(headers, header);
    return this.maybeSpan(`temporal:startWorkflow:${input.workflowType}`, () => next({ ...input, headers }), {
      workflowId: input.options.workflowId,
    });
  }

  async signal(input: WorkflowSignalInput, next: NextSignal): Promise<void> {
    const header = currentAgentsSpanHeader();
    if (!header) return next(input);

    const headers = injectAgentsTraceHeader(input.headers, header);
    return this.maybeSpan('temporal:signalWorkflow', () => next({ ...input, headers }), {
      workflowId: input.workflowExecution.workflowId,
      signalName: input.signalName,
    });
  }

  async query(input: WorkflowQueryInput, next: NextQuery): Promise<unknown> {
    const header = currentAgentsSpanHeader();
    if (!header) return next(input);

    const headers = injectAgentsTraceHeader(input.headers, header);
    return this.maybeSpan('temporal:queryWorkflow', () => next({ ...input, headers }), {
      workflowId: input.workflowExecution.workflowId,
      queryType: input.queryType,
    });
  }

  async startUpdate(input: WorkflowStartUpdateInput, next: NextStartUpdate): Promise<WorkflowStartUpdateOutput> {
    const header = currentAgentsSpanHeader();
    if (!header) return next(input);

    const headers = injectAgentsTraceHeader(input.headers, header);
    return this.maybeSpan('temporal:updateWorkflow', () => next({ ...input, headers }), {
      workflowId: input.workflowExecution.workflowId,
      updateName: input.updateName,
    });
  }

  async signalWithStart(input: WorkflowSignalWithStartInput, next: NextSignalWithStart): Promise<string> {
    // Always inject config header on signalWithStart (starts a workflow)
    let headers = injectAgentsConfigHeader(input.headers, this.configHeader);

    const header = currentAgentsSpanHeader();
    if (!header) return next({ ...input, headers });

    headers = injectAgentsTraceHeader(headers, header);
    return this.maybeSpan(`temporal:signalWithStartWorkflow:${input.workflowType}`, () => next({ ...input, headers }), {
      signalName: input.signalName,
    });
  }

  async startUpdateWithStart(
    input: WorkflowStartUpdateWithStartInput,
    next: NextStartUpdateWithStart
  ): Promise<WorkflowStartUpdateWithStartOutput> {
    // Always inject config header on startUpdateWithStart (starts a workflow)
    let workflowStartHeaders = injectAgentsConfigHeader(input.workflowStartHeaders, this.configHeader);

    const header = currentAgentsSpanHeader();
    if (!header) return next({ ...input, workflowStartHeaders });

    workflowStartHeaders = injectAgentsTraceHeader(workflowStartHeaders, header);
    const updateHeaders = injectAgentsTraceHeader(input.updateHeaders, header);
    return this.maybeSpan(
      `temporal:startUpdateWithStart:${input.workflowType}`,
      () => next({ ...input, workflowStartHeaders, updateHeaders }),
      { workflowType: input.workflowType, updateName: input.updateName }
    );
  }
}
