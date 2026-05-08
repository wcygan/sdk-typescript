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
import { addTraceProcessor } from '@openai/agents-core';
import { BaseAgentTracingProcessor, type SpanEntry } from '../common/base-tracing-processor';

/**
 * Bridges OpenAI Agents SDK trace events to OpenTelemetry spans in the
 * activity (worker-host) process. Uses a flat span map since each activity
 * execution is independent — no workflow-level isolation is needed.
 *
 * OTel trace and span IDs are derived deterministically from agent SDK IDs
 * using the same conversion functions as the workflow-side processor.
 * A {@link TemporalIdGenerator} controls each `tracer.startSpan()` call,
 * ensuring activity-side spans share the same OTel trace ID and form
 * correct parent-child links with workflow-side spans.
 */
class ActivityTracingProcessor extends BaseAgentTracingProcessor {
  private readonly spans = new Map<string, SpanEntry>();

  protected getEntry(id: string): SpanEntry | undefined {
    return this.spans.get(id);
  }

  protected setEntry(id: string, entry: SpanEntry): void {
    this.spans.set(id, entry);
  }

  protected deleteEntry(id: string): void {
    this.spans.delete(id);
  }

  protected allEntries(): Iterable<SpanEntry> {
    return this.spans.values();
  }

  protected clearAllEntries(): void {
    this.spans.clear();
  }
}

// Module-level idempotency flag — replaces the previous
// `globalThis[Symbol.for('temporal-openai-agents-activity-processor-registered')]`.
let activityProcessorRegistered = false;

/**
 * Registers an {@link ActivityTracingProcessor} on the activity-process global
 * agent SDK trace provider. Idempotent — first call wins; subsequent calls are no-ops.
 *
 * Called by {@link OpenAIAgentsTraceActivityInboundInterceptor} on first activity
 * execution so that `withCustomSpan('temporal:executeActivity', ...)` produces
 * OTel spans via the registered processor.
 */
export function ensureActivityTracingProcessorRegistered(): void {
  if (activityProcessorRegistered) return;
  activityProcessorRegistered = true;

  addTraceProcessor(new ActivityTracingProcessor());
}
