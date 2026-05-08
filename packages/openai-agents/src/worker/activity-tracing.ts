/**
 * Activity-side OpenAI Agents SDK to OTel bridge.
 *
 * Activities need their own TracingProcessor because the workflow-side
 * TemporalTracingProcessor runs inside the V8 sandbox and uses
 * workflow-scoped state. This processor uses a flat span map —
 * each activity execution is independent. Registered idempotently
 * by the activity interceptor so that `withCustomSpan` calls produce
 * OTel spans.
 */
import * as otel from '@opentelemetry/api';
import {
  type TracingProcessor,
  type Span,
  type Trace,
  type SpanData,
  addTraceProcessor,
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
} from '../common/tracing-bridge';

interface SpanEntry {
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

/**
 * Bridges OpenAI Agents SDK trace events to OpenTelemetry spans in the
 * activity (worker-host) process. Unlike the workflow-side processor,
 * this uses a flat span map — each activity execution is independent.
 *
 * OTel trace and span IDs are derived deterministically from agent SDK IDs
 * using the same conversion functions as the workflow-side processor.
 * A {@link TemporalIdGenerator} controls each `tracer.startSpan()` call,
 * ensuring activity-side spans share the same OTel trace ID and form
 * correct parent-child links with workflow-side spans.
 */
class ActivityTracingProcessor implements TracingProcessor {
  private readonly tracer: otel.Tracer;
  private readonly idGen: TemporalIdGenerator;
  private readonly spans = new Map<string, SpanEntry>();

  constructor() {
    this.tracer = otel.trace.getTracer(TRACER_NAME);
    this.idGen = installTemporalIdGenerator(this.tracer);
  }

  async onTraceStart(trace: Trace): Promise<void> {
    const attrs: otel.Attributes = { 'openai.agents.trace_id': trace.traceId };
    if (trace.name) attrs['openai.agents.trace.name'] = trace.name;
    if (trace.groupId) attrs['openai.agents.trace.group_id'] = trace.groupId;

    const otelTraceId = agentTraceIdToOtelTraceId(trace.traceId);
    const rootOtelSpanId = otelTraceId.slice(0, 16);

    this.idGen.seedTraceId(otelTraceId);
    this.idGen.seedSpanId(rootOtelSpanId);

    const span = this.tracer.startSpan('openai.agents.run', { attributes: attrs, root: true });
    const ctx = otel.trace.setSpan(otel.ROOT_CONTEXT, span);
    this.spans.set(trace.traceId, { span, context: ctx });
  }

  async onTraceEnd(trace: Trace): Promise<void> {
    const entry = this.spans.get(trace.traceId);
    if (!entry) return;
    entry.span.setStatus({ code: otel.SpanStatusCode.OK });
    entry.span.end();
    this.spans.delete(trace.traceId);
  }

  async onSpanStart(span: Span<SpanData>): Promise<void> {
    const data = span.spanData;
    const name = spanNameFromData(data);
    const attrs = staticAttributesFromSpanData(data);

    this.idGen.seedSpanId(agentSpanIdToOtelSpanId(span.spanId));

    let parentCtx: otel.Context;
    const parentEntry = span.parentId ? this.spans.get(span.parentId) : this.spans.get(span.traceId);
    if (parentEntry) {
      parentCtx = parentEntry.context;
    } else {
      // The parent span lives in the workflow-side processor. Derive a
      // synthetic OTel parent from the agent SDK IDs so this span joins
      // the same OTel trace tree with the correct parent link.
      const otelTraceId = agentTraceIdToOtelTraceId(span.traceId);
      const otelParentId = span.parentId ? agentSpanIdToOtelSpanId(span.parentId) : otelTraceId.slice(0, 16);
      parentCtx = syntheticParentContext(otelTraceId, otelParentId);
    }

    const otelSpan = this.tracer.startSpan(name, { attributes: attrs }, parentCtx);
    const ctx = otel.trace.setSpan(parentCtx, otelSpan);
    this.spans.set(span.spanId, { span: otelSpan, context: ctx });
  }

  async onSpanEnd(span: Span<SpanData>): Promise<void> {
    const entry = this.spans.get(span.spanId);
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
    this.spans.delete(span.spanId);
  }

  async shutdown(): Promise<void> {
    for (const [, entry] of this.spans) {
      entry.span.end();
    }
    this.spans.clear();
  }

  async forceFlush(): Promise<void> {
    // No buffering — spans are forwarded to OTel immediately
  }
}

const ACTIVITY_REGISTERED_KEY = Symbol.for('temporal-openai-agents-activity-processor-registered');

/**
 * Registers an {@link ActivityTracingProcessor} on the activity-process global
 * agent SDK trace provider. Idempotent — first call wins; subsequent calls are no-ops.
 *
 * Called by {@link OpenAIAgentsTraceActivityInboundInterceptor} on first activity
 * execution so that `withCustomSpan('temporal:executeActivity', ...)` produces
 * OTel spans via the registered processor.
 */
export function ensureActivityTracingProcessorRegistered(): void {
  if ((globalThis as any)[ACTIVITY_REGISTERED_KEY]) return;
  (globalThis as any)[ACTIVITY_REGISTERED_KEY] = true;

  addTraceProcessor(new ActivityTracingProcessor());
}
