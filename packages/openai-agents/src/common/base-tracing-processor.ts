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
  TemporalIdGenerator,
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

  /**
   * @param idGenerator - If provided, used directly (workflow side creates
   *   its own). If omitted, the constructor reads the global OTel
   *   TracerProvider — if it exposes a `temporalIdGenerator` getter
   *   (i.e. it's a `ReplaySafeTracerProvider`), uses that generator.
   *   If no provider is registered (default no-op), falls back to a
   *   default `TemporalIdGenerator`. If a non-`ReplaySafeTracerProvider`
   *   was explicitly registered (e.g. a plain `BasicTracerProvider`),
   *   throws a descriptive error.
   */
  constructor(idGenerator?: TemporalIdGenerator) {
    if (idGenerator) {
      this.idGen = idGenerator;
    } else {
      // Activity/host side: try to read from the global provider.
      // OTel API wraps the real provider in a ProxyTracerProvider; unwrap it.
      const raw = otel.trace.getTracerProvider();
      const provider = typeof (raw as any).getDelegate === 'function' ? (raw as any).getDelegate() : raw;
      if (provider && typeof (provider as any).temporalIdGenerator !== 'undefined') {
        // ReplaySafeTracerProvider — use its generator.
        this.idGen = (provider as any).temporalIdGenerator;
      } else if (provider && typeof (provider as any).addSpanProcessor === 'function') {
        // A real TracerProvider (e.g. BasicTracerProvider) was registered
        // but it's not a ReplaySafeTracerProvider. This is a user error.
        throw new Error(
          '@temporalio/openai-agents: the global TracerProvider must be a ReplaySafeTracerProvider. ' +
            'Use `createTracerProvider()` from @temporalio/openai-agents and pass it to ' +
            '`trace.setGlobalTracerProvider(...)` before initializing the plugin. ' +
            'See the package README for the wiring pattern.'
        );
      } else {
        // No real provider registered (default no-op). Fall back silently.
        this.idGen = new TemporalIdGenerator();
      }
    }
    this.tracer = otel.trace.getTracer(TRACER_NAME);
  }

  /**
   * Look up a span entry by its agent SDK span/trace ID. Subclasses
   * determine the scoping strategy.
   */
  protected abstract getEntry(spanId: string): SpanEntry | undefined;

  /**
   * Store a span entry. Subclasses determine the scoping strategy.
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
