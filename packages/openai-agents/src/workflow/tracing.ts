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

// --- Existing public helpers (preserved) ---

export function isInWorkflow(): boolean {
  return inWorkflowContext();
}

export function isReplaying(): boolean {
  if (!inWorkflowContext()) return false;
  return workflowInfo().unsafe.isReplaying;
}

// --- OTel bridge: maps OpenAI Agents SDK trace events to OTel spans ---

const TRACER_NAME = '@temporalio/openai-agents';
const REGISTERED_KEY = Symbol.for('temporal-openai-agents-processor-registered');

// Shared config symbol — read by workflow interceptors (trace-interceptor.ts)
// to gate addTemporalSpans / startTraces behavior.
const CONFIG_SYMBOL = Symbol.for('temporal-openai-agents-config');

function spanNameFromData(data: SpanData): string {
  switch (data.type) {
    case 'agent':
      return `openai.agents.agent:${data.name}`;
    case 'function':
      return `openai.agents.function:${data.name}`;
    case 'generation':
      return 'openai.agents.generation';
    case 'response':
      return 'openai.agents.response';
    case 'handoff':
      return 'openai.agents.handoff';
    case 'guardrail':
      return `openai.agents.guardrail:${data.name}`;
    case 'custom':
      return `openai.agents.custom:${data.name}`;
    case 'transcription':
      return 'openai.agents.transcription';
    case 'speech':
      return 'openai.agents.speech';
    case 'speech_group':
      return 'openai.agents.speech_group';
    case 'mcp_tools':
      return 'openai.agents.mcp_tools';
    default:
      return 'openai.agents.unknown';
  }
}

function staticAttributesFromSpanData(data: SpanData): otel.Attributes {
  const attrs: otel.Attributes = { 'openai.agents.span_type': data.type };
  switch (data.type) {
    case 'agent':
      attrs['openai.agents.agent.name'] = data.name;
      if (data.handoffs) attrs['openai.agents.agent.handoffs'] = data.handoffs.join(',');
      if (data.output_type) attrs['openai.agents.agent.output_type'] = data.output_type;
      break;
    case 'function':
      attrs['openai.agents.function.name'] = data.name;
      break;
    case 'handoff':
      if (data.from_agent) attrs['openai.agents.handoff.from_agent'] = data.from_agent;
      if (data.to_agent) attrs['openai.agents.handoff.to_agent'] = data.to_agent;
      break;
    case 'guardrail':
      attrs['openai.agents.guardrail.name'] = data.name;
      break;
    case 'custom':
      attrs['openai.agents.custom.name'] = data.name;
      break;
    case 'mcp_tools':
      if (data.server) attrs['openai.agents.mcp_tools.server'] = data.server;
      break;
  }
  return attrs;
}

function dynamicAttributesFromSpanData(data: SpanData): otel.Attributes {
  const attrs: otel.Attributes = {};
  switch (data.type) {
    case 'agent':
      if (data.tools) attrs['openai.agents.agent.tools'] = data.tools.join(',');
      break;
    case 'generation':
      if (data.model) attrs['openai.agents.generation.model'] = data.model;
      break;
    case 'guardrail':
      attrs['openai.agents.guardrail.triggered'] = data.triggered;
      break;
    case 'mcp_tools':
      if (data.result) attrs['openai.agents.mcp_tools.result'] = data.result.join(',');
      break;
  }
  return attrs;
}

interface SpanEntry {
  span: otel.Span;
  context: otel.Context;
}

export interface TemporalTracingProcessorOptions {
  /**
   * When `true`, emit OTel spans even during workflow replay. Defaults to `false`.
   * Useful for debugging replay-divergence issues where trace output helps identify
   * which spans differ between original execution and replay.
   */
  startSpansInReplay?: boolean;

  /**
   * When `true` (default), workflow/activity interceptors wrap calls in
   * `temporal:*` custom spans for Temporal-specific instrumentation.
   * Set to `false` to disable these spans while keeping trace propagation.
   *
   * Stored on `globalThis` so workflow interceptors (loaded via workflowModules)
   * can read it without direct access to plugin options.
   *
   * Mirrors Python's `add_temporal_spans` parameter.
   */
  addTemporalSpans?: boolean;

  /**
   * When `true`, restored trace contexts fire processor events (`onTraceStart`,
   * `onSpanStart`). When `false` (default), sets ALS context directly.
   *
   * Stored on `globalThis` for workflow interceptor access.
   * Mirrors Python's `start_traces` parameter.
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
 * **Timestamps**: The Temporal V8 sandbox replaces `Date` with a deterministic
 * clock. Upstream's only clock source is `timeIso()` in
 * `@openai/agents-core/dist/tracing/utils.js`, which calls
 * `new Date().toISOString()` — verified against v0.3.9. No `performance.now`,
 * `hrtime`, or other clock APIs are used. If a future upstream version
 * introduces a non-Date clock source, timestamps would diverge on replay
 * and an explicit override (like `installDeterministicTraceIds` for IDs)
 * would be needed.
 */
export class TemporalTracingProcessor implements TracingProcessor {
  private readonly tracer: otel.Tracer;
  private readonly startSpansInReplay: boolean;
  private readonly spans = new Map<string, Map<string, SpanEntry>>();

  constructor(options?: TemporalTracingProcessorOptions) {
    this.tracer = otel.trace.getTracer(TRACER_NAME);
    this.startSpansInReplay = options?.startSpansInReplay ?? false;
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

  private shouldSkip(): boolean {
    return !this.startSpansInReplay && isReplaying();
  }

  async onTraceStart(trace: Trace): Promise<void> {
    if (this.shouldSkip()) return;

    const parentCtx = otel.context.active();
    const attrs: otel.Attributes = { 'openai.agents.trace_id': trace.traceId };
    if (trace.name) attrs['openai.agents.trace.name'] = trace.name;
    if (trace.groupId) attrs['openai.agents.trace.group_id'] = trace.groupId;

    const span = this.tracer.startSpan('openai.agents.run', { attributes: attrs }, parentCtx);
    const ctx = otel.trace.setSpan(parentCtx, span);
    this.getWorkflowSpans().set(trace.traceId, { span, context: ctx });
  }

  async onTraceEnd(trace: Trace): Promise<void> {
    if (this.shouldSkip()) return;

    const entry = this.getSpanEntry(trace.traceId);
    if (!entry) return;
    entry.span.setStatus({ code: otel.SpanStatusCode.OK });
    entry.span.end();
    this.deleteSpanEntry(trace.traceId);
  }

  async onSpanStart(span: Span<SpanData>): Promise<void> {
    if (this.shouldSkip()) return;

    const data = span.spanData;
    const name = spanNameFromData(data);
    const attrs = staticAttributesFromSpanData(data);

    let parentCtx: otel.Context;
    const parentEntry = span.parentId ? this.getSpanEntry(span.parentId) : this.getSpanEntry(span.traceId);
    if (parentEntry) {
      parentCtx = parentEntry.context;
    } else {
      parentCtx = otel.context.active();
    }

    const otelSpan = this.tracer.startSpan(name, { attributes: attrs }, parentCtx);
    const ctx = otel.trace.setSpan(parentCtx, otelSpan);
    this.getWorkflowSpans().set(span.spanId, { span: otelSpan, context: ctx });
  }

  async onSpanEnd(span: Span<SpanData>): Promise<void> {
    if (this.shouldSkip()) return;

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
 *
 * Mirrors Python's approach of overriding `gen_trace_id`/`gen_span_id` on
 * the trace provider.
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
