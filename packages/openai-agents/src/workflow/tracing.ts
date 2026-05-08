import { addTraceProcessor, getGlobalTraceProvider } from '@openai/agents-core';
import { inWorkflowContext, workflowInfo, uuid4, log } from '@temporalio/workflow';
import { BaseAgentTracingProcessor, type SpanEntry } from '../common/base-tracing-processor';

// --- Workflow context helpers ---

export function isInWorkflow(): boolean {
  return inWorkflowContext();
}

export function isReplaying(): boolean {
  if (!inWorkflowContext()) return false;
  return workflowInfo().unsafe.isReplaying;
}

// --- OTel bridge: maps OpenAI Agents SDK trace events to OTel spans ---

const REGISTERED_KEY = Symbol.for('temporal-openai-agents-processor-registered');

// Shared config symbol — read by workflow interceptors (trace-interceptor.ts)
// to gate addTemporalSpans / startTraces behavior.
const CONFIG_SYMBOL = Symbol.for('temporal-openai-agents-config');

export interface TemporalTracingProcessorOptions {
  /**
   * When `true`, workflow/activity interceptors wrap calls in
   * `temporal:*` custom spans for Temporal-specific instrumentation.
   * Set to `false` (default) to disable these spans while keeping trace propagation.
   *
   * Stored on `globalThis` so workflow interceptors (loaded via workflowModules)
   * can read it without direct access to plugin options.
   *
   * Default: `false`.
   */
  addTemporalSpans?: boolean;

  /**
   * When `true`, restored trace contexts fire processor events (`onTraceStart`,
   * `onSpanStart`). When `false` (default), sets ALS context directly.
   * Stored on `globalThis` for workflow interceptor access.
   */
  startTraces?: boolean;
}

/**
 * Bridges OpenAI Agents SDK trace events to OpenTelemetry spans.
 *
 * Deterministic trace/span IDs are ensured via two layers:
 * 1. **Explicit override** (primary): `installDeterministicTraceIds()` wraps
 *    the upstream TraceProvider's `createTrace`/`createSpan` methods to inject
 *    IDs from `workflow.uuid4()` — a per-workflow seeded PRNG. This is immune
 *    to upstream changes in ID generation strategy.
 * 2. **Polyfill** (belt-and-suspenders): `load-polyfills.ts` replaces
 *    `crypto.randomUUID` with `uuid4()`. Catches any remaining `randomUUID`
 *    calls from upstream that bypass the TraceProvider.
 *
 * OTel trace/span IDs are derived deterministically from agent SDK IDs
 * using {@link agentTraceIdToOtelTraceId} and {@link agentSpanIdToOtelSpanId}.
 * A {@link TemporalIdGenerator} is installed on the OTel Tracer to ensure
 * each `tracer.startSpan()` produces spans with those derived IDs.
 * The activity-side processor applies the same conversion, so OTel spans
 * from both sides share the same trace ID and form a single trace tree.
 *
 * **Timestamps**: The Temporal V8 sandbox replaces `Date` with a deterministic
 * clock. Upstream's only clock source is `timeIso()` in
 * `@openai/agents-core/dist/tracing/utils.js`, which calls
 * `new Date().toISOString()`. No `performance.now`, `hrtime`, or other clock
 * APIs are used. If a future upstream version introduces a non-Date clock
 * source, timestamps would diverge on replay and an explicit override
 * (like `installDeterministicTraceIds` for IDs) would be needed.
 */
export class TemporalTracingProcessor extends BaseAgentTracingProcessor {
  /**
   * Spans are keyed by `(workflowId, spanId)` so that concurrent workflows
   * sharing a single V8 isolate under `reuseV8Context: true` never leak
   * span state across runs.
   */
  private readonly spans = new Map<string, Map<string, SpanEntry>>();

  constructor(_options?: TemporalTracingProcessorOptions) {
    super();
  }

  private getWorkflowSpans(): Map<string, SpanEntry> {
    const wfId = workflowInfo().workflowId;
    let inner = this.spans.get(wfId);
    if (!inner) {
      inner = new Map();
      this.spans.set(wfId, inner);
    }
    return inner;
  }

  protected getEntry(id: string): SpanEntry | undefined {
    return this.spans.get(workflowInfo().workflowId)?.get(id);
  }

  protected setEntry(id: string, entry: SpanEntry): void {
    this.getWorkflowSpans().set(id, entry);
  }

  protected deleteEntry(id: string): void {
    const wfId = workflowInfo().workflowId;
    const inner = this.spans.get(wfId);
    if (!inner) return;
    inner.delete(id);
    if (inner.size === 0) this.spans.delete(wfId);
  }

  protected *allEntries(): Iterable<SpanEntry> {
    for (const [, inner] of this.spans) {
      for (const [, entry] of inner) {
        yield entry;
      }
    }
  }

  protected clearAllEntries(): void {
    this.spans.clear();
  }
}

// --- Deterministic ID override ---

const IDS_INSTALLED_KEY = Symbol.for('temporal-openai-agents-deterministic-ids');

/**
 * Wraps the upstream TraceProvider's `createTrace` and `createSpan` methods
 * to inject deterministic IDs from `workflow.uuid4()`. This is the primary
 * defense against replay non-determinism in trace/span IDs — the polyfill
 * in `load-polyfills.ts` is a secondary safety net.
 */
function installDeterministicTraceIds(): void {
  if ((globalThis as any)[IDS_INSTALLED_KEY]) return;
  (globalThis as any)[IDS_INSTALLED_KEY] = true;

  const provider = getGlobalTraceProvider() as any;

  const origCreateTrace = provider.createTrace.bind(provider);
  provider.createTrace = (options: any) => {
    return origCreateTrace({
      ...options,
      traceId: options.traceId ?? `trace_${uuid4().replace(/-/g, '')}`,
    });
  };

  const origCreateSpan = provider.createSpan.bind(provider);
  provider.createSpan = (options: any, parent?: any) => {
    return origCreateSpan(
      {
        ...options,
        spanId: options.spanId ?? `span_${uuid4().replace(/-/g, '').slice(0, 24)}`,
      },
      parent
    );
  };
}

/**
 * Appends a {@link TemporalTracingProcessor} to the OpenAI Agents SDK's
 * global processor list, enables tracing, installs deterministic ID
 * generation, and stores interceptor config on `globalThis`.
 *
 * **Side effects**:
 * - Mutates the upstream TraceProvider (processor list + createTrace/createSpan)
 * - Sets `globalThis[Symbol.for('temporal-openai-agents-config')]`
 *
 * Called automatically by the {@link TemporalOpenAIRunner} constructor.
 *
 * Idempotent — first call wins. Subsequent calls with different options
 * log a warning but are otherwise no-ops.
 */
export function ensureTracingProcessorRegistered(options?: TemporalTracingProcessorOptions): void {
  if ((globalThis as any)[REGISTERED_KEY]) {
    if (options) {
      log.warn(
        'ensureTracingProcessorRegistered called again with options — first-call options win, this call ignored'
      );
    }
    return;
  }
  (globalThis as any)[REGISTERED_KEY] = true;

  // Store interceptor config for workflow interceptors (loaded via workflowModules)
  (globalThis as any)[CONFIG_SYMBOL] = {
    addTemporalSpans: options?.addTemporalSpans,
    startTraces: options?.startTraces,
  };

  addTraceProcessor(new TemporalTracingProcessor(options));
  installDeterministicTraceIds();
}
