import {
  Agent,
  Runner,
  type AgentOutputType,
  type CallModelInputFilter,
  type HandoffInputData,
  type InputGuardrail,
  type ModelSettings,
  type OutputGuardrail,
  type RunResult,
  type Session,
  type SessionInputCallback,
  type TracingConfig,
} from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import { workflowInfo } from '@temporalio/workflow';
import { DEFAULT_MODEL_ACTIVITY_OPTIONS, type ModelActivityOptions } from '../common/model-activity-options';
import { unwrapTemporalFailure } from '../common/errors';
import { PlaceholderModelProvider } from './dummy-model-provider';
import { convertAgent } from './convert-agent';
import { ensureTracingProcessorRegistered } from './tracing';
import { getOriginalPluginConfig, setPluginConfig } from './plugin-config-store';

export interface TemporalRunOptions<TContext = undefined> {
  /** Run context passed to agents and tools */
  context?: TContext;
  /** Maximum agent loop turns before aborting */
  maxTurns?: number;
  /** Previous OpenAI response ID for conversation continuity */
  previousResponseId?: string;
  /** OpenAI conversation ID for multi-turn persistence */
  conversationId?: string;
  /** Session state for conversation memory */
  session?: Session;
  /** Customize how session history merges with current turn input */
  sessionInputCallback?: SessionInputCallback;
  /** Edit system instructions or input items just before calling the model */
  callModelInputFilter?: CallModelInputFilter;
  /** Per-run tracing config override */
  tracing?: TracingConfig;
  // signal intentionally omitted — use Temporal CancellationScope for workflow cancellation

  /** Runner-level config overrides */
  runConfig?: {
    /** Model name override (string only — Model objects can't cross the workflow/activity boundary) */
    model?: string;
    /** Global model settings (temperature, topP, etc.). Non-null values override agent-specific settings. */
    modelSettings?: ModelSettings;
    /** Global handoff input filter. Agent-level inputFilter takes precedence. */
    handoffInputFilter?: (input: HandoffInputData) => HandoffInputData;
    /** Input guardrails run inline in the workflow — callbacks must be deterministic */
    inputGuardrails?: InputGuardrail[];
    /** Output guardrails run inline in the workflow — callbacks must be deterministic */
    outputGuardrails?: OutputGuardrail<AgentOutputType<unknown>>[];
    /** Disable tracing for this run */
    tracingDisabled?: boolean;
    /** Include sensitive data (tool I/O, LLM outputs) in trace spans */
    traceIncludeSensitiveData?: boolean;
    /** Logical name for the run, used in tracing */
    workflowName?: string;
    /** Custom trace ID */
    traceId?: string;
    /** Grouping ID for linking traces (e.g., chat thread ID) */
    groupId?: string;
    /** Additional metadata attached to the trace */
    traceMetadata?: Record<string, string>;
  };
}

export interface TemporalOpenAIRunnerOptions {
  /**
   * Model activity options (timeouts, retry, task queue, etc.).
   * Overrides per-field any `modelParams` propagated from the plugin via header.
   * When omitted, defaults are merged from the *original* plugin config
   * (header-injected by the plugin client interceptor on workflow start).
   * Subsequent `TemporalOpenAIRunner` constructions in the same workflow do not
   * inherit overrides from earlier ones — the original config is the merge base.
   *
   * **Merge is shallow** (object spread): a runner-level
   * `retryPolicy: { initialInterval: '1s' }` *replaces* a plugin-level
   * `retryPolicy: { maximumAttempts: 5 }` rather than deep-merging the two.
   * This matches Temporal SDK convention for activity options.
   *
   * Unlike the plugin-side `modelParams` ({@link SerializableModelActivityOptions}),
   * the runner accepts the full {@link ModelActivityOptions} — including the
   * function form `summaryOverride: ModelSummaryProvider`. This is the documented
   * escape hatch for dynamic summaries that need workflow-local context.
   * See {@link OpenAIAgentsPluginOptions.modelParams} and
   * {@link OpenAIAgentsTraceClientInterceptorOptions.modelParams} for the
   * serializable (plugin-side) counterpart.
   */
  modelParams?: ModelActivityOptions;

  /**
   * When `true`, workflow/activity interceptors wrap calls in
   * `temporal:*` custom spans for Temporal-specific instrumentation.
   * Set to `false` (default) to disable these spans while keeping trace propagation.
   * Overrides the plugin's `interceptorOptions.addTemporalSpans` for this workflow.
   *
   * Default: `false`.
   */
  addTemporalSpans?: boolean;

  /**
   * When `true`, restored trace contexts fire processor events (`onTraceStart`,
   * `onSpanStart`). When `false` (default), sets ALS context directly.
   * Overrides the plugin's `interceptorOptions.startTraces` for this workflow.
   */
  startTraces?: boolean;
}

/**
 * A Temporal-aware agent runner that delegates model calls to activities.
 *
 * Streaming is not supported in Temporal workflows because activities are
 * request-response. Use run() for all agent invocations.
 */
export class TemporalOpenAIRunner {
  private readonly modelParams: ModelActivityOptions;

  constructor(options?: TemporalOpenAIRunnerOptions) {
    // Register the tracing processor idempotently (no options — config lives in the store)
    ensureTracingProcessorRegistered();

    // Merge config: runner constructor args > original header config > defaults.
    // Reads from the *original* config (set once by the inbound interceptor),
    // NOT the effective config (which may have been overwritten by a prior
    // runner construction). This prevents modelParams accumulation across
    // multiple `new TemporalOpenAIRunner(...)` calls in the same workflow.
    const wfId = workflowInfo().workflowId;
    const fromOriginal = getOriginalPluginConfig(wfId);

    const mergedModelParams: ModelActivityOptions = {
      ...DEFAULT_MODEL_ACTIVITY_OPTIONS,
      ...fromOriginal?.modelParams,
      ...options?.modelParams,
    };

    setPluginConfig(wfId, {
      addTemporalSpans: options?.addTemporalSpans ?? fromOriginal?.addTemporalSpans ?? false,
      startTraces: options?.startTraces ?? fromOriginal?.startTraces ?? false,
      modelParams: mergedModelParams,
    });

    this.modelParams = mergedModelParams;
  }

  /**
   * Run an agent in workflow context. Model calls are delegated to activities
   * via ActivityBackedModel, while the agent loop runs durably in the workflow.
   */
  async run<TAgent extends Agent<any, any>, TContext = undefined>(
    agent: TAgent,
    input: string,
    options?: TemporalRunOptions<TContext>
  ): Promise<RunResult<any, TAgent>> {
    const { model: modelOverride, ...runnerConfigOverrides } = options?.runConfig ?? {};

    const converted = convertAgent(agent, this.modelParams, undefined, modelOverride);

    const innerRunner = new Runner({
      modelProvider: new PlaceholderModelProvider(),
      ...runnerConfigOverrides,
    });

    try {
      return (await innerRunner.run(converted, input, {
        maxTurns: options?.maxTurns,
        context: options?.context,
        previousResponseId: options?.previousResponseId,
        conversationId: options?.conversationId,
        session: options?.session,
        sessionInputCallback: options?.sessionInputCallback,
        callModelInputFilter: options?.callModelInputFilter,
        tracing: options?.tracing,
      })) as RunResult<any, TAgent>;
    } catch (error) {
      const temporalFailure = unwrapTemporalFailure(error);
      if (temporalFailure) throw temporalFailure;
      if (error instanceof Error) {
        throw ApplicationFailure.create({
          message: `Agent workflow failed: ${error.message}`,
          type: 'AgentsWorkflowError',
          nonRetryable: true,
          cause: error,
        });
      }
      throw error;
    }
  }
}
