import * as otel from '@opentelemetry/api';
import {
  type TracingProcessor,
  type Span,
  type Trace,
  type SpanData,
  addTraceProcessor,
  setTracingDisabled,
  getGlobalTraceProvider,
} from '@openai/agents-core';
import { inWorkflowContext, workflowInfo, uuid4, log } from '@temporalio/workflow';
import {
  TRACER_NAME,
  spanNameFromData,
  staticAttributesFromSpanData,
  dynamicAttributesFromSpanData,
  agentTraceIdToOtelTraceId,
  agentSpanIdToOtelSpanId,
  installSeedableIdGenerator,
  type SeedableIdGenerator,
} from '../common/tracing-bridge';

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

interface SpanEntry {
  span: otel.Span;
  context: otel.Context;
}

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
 * Builds an OTel context with a synthetic remote SpanContext for
 * establishing trace ID inheritance and parent-child links.
 */
function syntheticParentContext(otelTraceId: string, otelParentSpanId: string): otel.Context {
  const spanContext: otel.SpanContext = {
    traceId: otelTraceId,
    spanId: otelParentSpanId,
    traceFlags: otel.TraceFlags.SAMPLED,
    isRemote: true,
  };
  return otel.trace.setSpanContext(otel.ROOT_CONTEXT, spanContext);
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
 * A {@link SeedableIdGenerator} is installed on the OTel Tracer to ensure
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
export class TemporalTracingProcessor implements TracingProcessor {
  private readonly tracer: otel.Tracer;
  private readonly idGen: SeedableIdGenerator;
  private readonly spans = new Map<string, Map<string, SpanEntry>>();

  constructor(_options?: TemporalTracingProcessorOptions) {
    this.tracer = otel.trace.getTracer(TRACER_NAME);
    this.idGen = installSeedableIdGenerator(this.tracer);
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

  private getSpanEntry(id: string): SpanEntry | undefined {
    return this.spans.get(workflowInfo().workflowId)?.get(id);
  }

  private deleteSpanEntry(id: string): void {
    const wfId = workflowInfo().workflowId;
    const inner = this.spans.get(wfId);
    if (!inner) return;
    inner.delete(id);
    if (inner.size === 0) this.spans.delete(wfId);
  }

  async onTraceStart(trace: Trace): Promise<void> {
    const attrs: otel.Attributes = { 'openai.agents.trace_id': trace.traceId };
    if (trace.name) attrs['openai.agents.trace.name'] = trace.name;
    if (trace.groupId) attrs['openai.agents.trace.group_id'] = trace.groupId;

    const otelTraceId = agentTraceIdToOtelTraceId(trace.traceId);
    // Use the first 16 chars of the derived trace ID as the root span's
    // OTel span ID. This value is stable and unique per agent trace.
    const rootOtelSpanId = otelTraceId.slice(0, 16);

    // Seed so that tracer.startSpan() produces this exact trace+span ID pair.
    this.idGen.seedTraceId(otelTraceId);
    this.idGen.seedSpanId(rootOtelSpanId);

    const span = this.tracer.startSpan('openai.agents.run', { attributes: attrs, root: true });
    const ctx = otel.trace.setSpan(otel.ROOT_CONTEXT, span);
    this.getWorkflowSpans().set(trace.traceId, { span, context: ctx });
  }

  async onTraceEnd(trace: Trace): Promise<void> {
    const entry = this.getSpanEntry(trace.traceId);
    if (!entry) return;
    entry.span.setStatus({ code: otel.SpanStatusCode.OK });
    entry.span.end();
    this.deleteSpanEntry(trace.traceId);
  }

  async onSpanStart(span: Span<SpanData>): Promise<void> {
    const data = span.spanData;
    const name = spanNameFromData(data);
    const attrs = staticAttributesFromSpanData(data);

    // Seed the OTel span ID with a deterministic value derived from the
    // agent SDK span ID. The activity-side processor uses the same
    // conversion, enabling cross-boundary parent-child links.
    this.idGen.seedSpanId(agentSpanIdToOtelSpanId(span.spanId));

    let parentCtx: otel.Context;
    const parentEntry = span.parentId ? this.getSpanEntry(span.parentId) : this.getSpanEntry(span.traceId);
    if (parentEntry) {
      parentCtx = parentEntry.context;
    } else {
      // Derive a synthetic OTel parent from the agent SDK IDs so that
      // spans whose parent lives in a different processor (e.g. across
      // the workflow/activity boundary) still land in the same OTel trace.
      const otelTraceId = agentTraceIdToOtelTraceId(span.traceId);
      const otelParentId = span.parentId ? agentSpanIdToOtelSpanId(span.parentId) : otelTraceId.slice(0, 16);
      parentCtx = syntheticParentContext(otelTraceId, otelParentId);
    }

    const otelSpan = this.tracer.startSpan(name, { attributes: attrs }, parentCtx);
    const ctx = otel.trace.setSpan(parentCtx, otelSpan);
    this.getWorkflowSpans().set(span.spanId, { span: otelSpan, context: ctx });
  }

  async onSpanEnd(span: Span<SpanData>): Promise<void> {
    const entry = this.getSpanEntry(span.spanId);
    if (!entry) return;

    const dynAttrs = dynamicAttributesFromSpanData(span.spanData);
    for (const [key, value] of Object.entries(dynAttrs)) {
      if (value !== undefined) entry.span.setAttribute(key, value);
    }

    if (span.error) {
      entry.span.setStatus({ code: otel.SpanStatusCode.ERROR, message: span.error.message });
      entry.span.recordException(new Error(span.error.message));
    } else {
      entry.span.setStatus({ code: otel.SpanStatusCode.OK });
    }

    entry.span.end();
    this.deleteSpanEntry(span.spanId);
  }

  async shutdown(): Promise<void> {
    for (const [, inner] of this.spans) {
      for (const [, entry] of inner) {
        entry.span.end();
      }
    }
    this.spans.clear();
  }

  async forceFlush(): Promise<void> {
    // No buffering — spans are forwarded to OTel immediately
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

  setTracingDisabled(false);
  addTraceProcessor(new TemporalTracingProcessor(options));
  installDeterministicTraceIds();
}
