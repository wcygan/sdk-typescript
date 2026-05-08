import { addTraceProcessor, getCurrentTrace, getGlobalTraceProvider, Trace } from '@openai/agents-core';
import { inWorkflowContext, workflowInfo, uuid4 } from '@temporalio/workflow';
import { BaseAgentTracingProcessor, type SpanEntry } from '../common/base-tracing-processor';
import { TemporalIdGenerator } from '../common/tracing-bridge';

// --- Workflow context helpers ---

export function isInWorkflow(): boolean {
  return inWorkflowContext();
}

export function isReplaying(): boolean {
  if (!inWorkflowContext()) return false;
  return workflowInfo().unsafe.isReplaying;
}

// --- OTel bridge: maps OpenAI Agents SDK trace events to OTel spans ---

// Module-level state is private to this module and lives once per V8 isolate,
// matching the old globalThis semantics with stronger typing. Under
// `reuseV8Context: true` the module is loaded once per isolate.
let processorRegistered = false;

/**
 * Bridges OpenAI Agents SDK trace events to OpenTelemetry spans.
 *
 * Deterministic trace/span IDs are ensured via two layers:
 * 1. **Explicit override** (primary): `installDeterministicTraceIds()` wraps
 *    the upstream TraceProvider's `createTrace`/`createSpan` methods to inject
 *    IDs from `workflow.uuid4()` — a per-workflow seeded PRNG. This is immune
 *    to upstream changes in ID generation strategy.
 * 2. **Polyfill** (belt-and-suspenders): `@temporalio/workflow/polyfills`
 *    (loaded by `load-polyfills.ts`) replaces `crypto.randomUUID` with
 *    `uuid4()`. Catches any remaining `randomUUID` calls from upstream that
 *    bypass the TraceProvider.
 *
 * OTel trace/span IDs are derived deterministically from agent SDK IDs
 * using {@link agentTraceIdToOtelTraceId} and {@link agentSpanIdToOtelSpanId}.
 * A {@link TemporalIdGenerator} is installed on the OTel Tracer to ensure
 * each `tracer.startSpan()` produces spans with those derived IDs.
 * The activity-side processor applies the same conversion, so OTel spans
 * from both sides share the same trace ID and form a single trace tree.
 *
 * **Timestamps**: The Temporal V8 sandbox replaces `Date` with a deterministic
 * clock, which covers upstream's timestamp generation. If a future upstream
 * version introduces a non-Date clock source (e.g. `performance.now`),
 * timestamps could diverge on replay and an explicit override would be needed.
 */
export class TemporalTracingProcessor extends BaseAgentTracingProcessor {
  /**
   * Spans are keyed by `(workflowId, spanId)` so that concurrent workflows
   * sharing a single V8 isolate under `reuseV8Context: true` never leak
   * span state across runs.
   */
  private readonly spans = new Map<string, Map<string, SpanEntry>>();

  constructor() {
    // The workflow V8 sandbox cannot import `@opentelemetry/sdk-trace-base`'s
    // `BasicTracerProvider`: the transitive dependency `@opentelemetry/core`
    // references the global `performance` object in its browser platform shim
    // (`platform/browser/performance.ts`), which does not exist in Temporal's
    // V8 sandbox, causing a `ReferenceError: performance is not defined` at
    // workflow module-load time.
    //
    // WARNING: Writing to a private OTel SDK field. If `BasicTracerProvider`
    // restructures or renames `_idGenerator` in a future major, this breaks
    // silently and traces fragment without erroring. Revisit if bundling of
    // `BasicTracerProvider` becomes possible in the workflow sandbox.
    const idGen = new TemporalIdGenerator();
    super(idGen);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (this.tracer as any)._idGenerator = idGen;
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

// Module-level idempotency flag — once per V8 isolate.
let deterministicIdsInstalled = false;

/**
 * Wraps the upstream TraceProvider's `createTrace` and `createSpan` methods
 * to inject deterministic IDs from `workflow.uuid4()`. This is the primary
 * defense against replay non-determinism in trace/span IDs — the polyfill
 * in `@temporalio/workflow/polyfills` is a secondary safety net.
 */
function installDeterministicTraceIds(): void {
  if (deterministicIdsInstalled) return;
  deterministicIdsInstalled = true;

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
 * global processor list and enables deterministic ID generation.
 *
 * **Side effects**:
 * - Mutates the upstream TraceProvider (processor list + createTrace/createSpan)
 *
 * Called automatically by the {@link TemporalOpenAIRunner} constructor.
 * Per-workflow config (addTemporalSpans, startTraces, modelParams) is stored
 * in the per-workflow plugin-config-store, NOT here.
 *
 * Idempotent — runs once per V8 isolate.
 */
export function ensureTracingProcessorRegistered(): void {
  if (processorRegistered) return;
  processorRegistered = true;

  addTraceProcessor(new TemporalTracingProcessor());
  installDeterministicTraceIds();
  verifyAlsContextShape();
}

// --- ALS context shape smoke check ---

const AGENTS_CORE_ALS_SYMBOL = Symbol.for('openai.agents.core.asyncLocalStorage');

/**
 * Verify that upstream's undocumented ALS store shape `{ trace, span, active }`
 * still works as expected. The plugin reaches into this shape to set trace
 * context without firing processor events (see common/trace-context.ts).
 *
 * Runs once per V8 isolate during the first {@link TemporalOpenAIRunner}
 * construction in that isolate (called from {@link ensureTracingProcessorRegistered}).
 * If upstream renames `active`, restructures the store, or changes how
 * `getCurrentTrace()` reads it, this fails loudly instead of silently
 * losing trace context at runtime.
 */
function verifyAlsContextShape(): void {
  // Force ALS initialization — getCurrentTrace() triggers lazy creation of
  // the AsyncLocalStorage instance on globalThis.
  getCurrentTrace();

  const als = (globalThis as any)[AGENTS_CORE_ALS_SYMBOL] as
    | { run: <R>(store: unknown, callback: () => R) => R }
    | undefined;

  if (!als) {
    // If ALS isn't available, the plugin's trace propagation is already
    // degraded (withContextOnly falls back to running fn directly). Not an
    // error — just nothing to verify.
    return;
  }

  const sentinel = new Trace({ traceId: 'smoke-check', name: 'smoke-check' });

  // Manually save/restore the ALS store: the workflow bundle picks an upstream shim
  // whose `run()` doesn't restore context, so the sentinel would otherwise persist
  // across subsequent workflow executions sharing the V8 isolate.
  //
  // With header-driven trace context restoration (Item 18), `prev` may be a
  // meaningful restored context from the inbound interceptor, not just undefined.
  // The save/restore is still correct — we don't want the sentinel to replace it.
  const prev = (als as any).getStore?.();
  let retrieved: unknown;
  try {
    retrieved = als.run({ trace: sentinel, span: undefined, active: true }, () => getCurrentTrace());
  } finally {
    if ((als as any).enterWith) (als as any).enterWith(prev);
  }

  if (retrieved !== sentinel) {
    throw new Error(
      "@temporalio/openai-agents: agent SDK ALS context shape has drifted. " +
        "The plugin reached into upstream's internal context shape (`{ trace, span, active }`); " +
        'one of those fields is no longer set or no longer load-bearing. ' +
        'Open an issue against @temporalio/openai-agents.'
    );
  }
}
