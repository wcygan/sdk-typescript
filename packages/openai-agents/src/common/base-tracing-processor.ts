/**
 * Abstract base class for OpenAI Agents SDK → OTel tracing processors.
 *
 * Handles the full lifecycle of mapping agent SDK trace/span events to
 * OpenTelemetry spans: `onTraceStart`, `onTraceEnd`, `onSpanStart`,
 * `onSpanEnd`, `shutdown`, and `forceFlush`. Subclasses provide only
 * the storage primitives (`getEntry` / `setEntry` / `deleteEntry` /
 * `allEntries`) so that each environment can use the appropriate map
 * structure.
 *
 * OTel trace/span IDs are derived deterministically from agent SDK IDs
 * using {@link agentTraceIdToOtelTraceId} and {@link agentSpanIdToOtelSpanId}.
 * A {@link TemporalIdGenerator} seeds each `tracer.startSpan()` call so
 * that the resulting OTel span carries the correct derived ID. Both the
 * workflow-side and activity-side processors use the same derivation,
 * unifying all spans into a single OTel trace tree.
 */
import * as otel from '@opentelemetry/api';
import {
  type TracingProcessor,
  type Span,
  type Trace,
  type SpanData,
} from '@openai/agents-core';
import {
  TRACER_NAME,
  spanNameFromData,
  staticAttributesFromSpanData,
  dynamicAttributesFromSpanData,
  agentTraceIdToOtelTraceId,
  agentSpanIdToOtelSpanId,
  installTemporalIdGenerator,
  type TemporalIdGenerator,
} from './tracing-bridge';

export interface SpanEntry {
  span: otel.Span;
  context: otel.Context;
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

export abstract class BaseAgentTracingProcessor implements TracingProcessor {
  protected readonly tracer: otel.Tracer;
  protected readonly idGen: TemporalIdGenerator;

  constructor() {
    this.tracer = otel.trace.getTracer(TRACER_NAME);
    this.idGen = installTemporalIdGenerator(this.tracer);
  }

  /**
   * Look up a span entry by its agent SDK span/trace ID.
   *
   * The workflow-side subclass keys spans by `(workflowId, spanId)` to
   * isolate concurrent workflows sharing the same V8 isolate under
   * `reuseV8Context: true`. The activity-side subclass uses a flat map
   * since each activity execution is independent.
   */
  protected abstract getEntry(spanId: string): SpanEntry | undefined;

  /**
   * Store a span entry.
   *
   * The workflow-side subclass nests entries under the current workflow ID
   * so that `reuseV8Context: true` isolates never leak spans across
   * concurrent workflow runs. The activity-side subclass stores entries in
   * a flat map.
   */
  protected abstract setEntry(spanId: string, entry: SpanEntry): void;

  /** Remove a span entry after the span has ended. */
  protected abstract deleteEntry(spanId: string): void;

  /** Iterate all live entries for `shutdown()` cleanup. */
  protected abstract allEntries(): Iterable<SpanEntry>;

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
    this.setEntry(trace.traceId, { span, context: ctx });
  }

  async onTraceEnd(trace: Trace): Promise<void> {
    const entry = this.getEntry(trace.traceId);
    if (!entry) return;
    entry.span.setStatus({ code: otel.SpanStatusCode.OK });
    entry.span.end();
    this.deleteEntry(trace.traceId);
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
    const parentEntry = span.parentId ? this.getEntry(span.parentId) : this.getEntry(span.traceId);
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
    this.setEntry(span.spanId, { span: otelSpan, context: ctx });
  }

  async onSpanEnd(span: Span<SpanData>): Promise<void> {
    const entry = this.getEntry(span.spanId);
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
    this.deleteEntry(span.spanId);
  }

  async shutdown(): Promise<void> {
    for (const entry of this.allEntries()) {
      entry.span.end();
    }
    this.clearAllEntries();
  }

  /** Remove all stored entries. Called by `shutdown()`. */
  protected abstract clearAllEntries(): void;

  async forceFlush(): Promise<void> {
    // No buffering — spans are forwarded to OTel immediately
  }
}
