import { getCurrentTrace, getCurrentSpan } from '@openai/agents-core';
import { defaultPayloadConverter, type Headers } from '@temporalio/common';

/** Header key under which agent trace context is propagated across Temporal boundaries. */
export const AGENTS_TRACE_HEADER_KEY = '__openai_span';

/** Wire shape of the propagated agent trace context. */
export interface AgentsSpanHeader {
  traceName: string;
  spanId: string | null;
  traceId: string | null;
}

/**
 * Serialize an AgentsSpanHeader into a Temporal headers map.
 *
 * Always uses `defaultPayloadConverter` — never a user-configurable converter.
 * SDK-injected headers (trace context, agent-SDK spans) must use the default
 * converter so that every participant in a Temporal cluster — clients, workers,
 * replayers — can decode them without coordinating codec config. The default
 * converter is the only converter every party is guaranteed to have. Allowing
 * override on either side is a footgun: a user codec that doesn't gracefully
 * fall through to default JSON for unknown payloads silently breaks header
 * propagation.
 */
export function injectAgentsTraceHeader(headers: Headers, info: AgentsSpanHeader): Headers {
  return { ...headers, [AGENTS_TRACE_HEADER_KEY]: defaultPayloadConverter.toPayload(info) };
}

/**
 * Deserialize an AgentsSpanHeader from a Temporal headers map; returns null if absent.
 *
 * Always uses `defaultPayloadConverter` — see {@link injectAgentsTraceHeader} for
 * the rationale. Both sides must use the same converter, and the default converter
 * is the only one every cluster participant is guaranteed to have.
 */
export function extractAgentsTraceHeader(headers: Headers): AgentsSpanHeader | null {
  const payload = headers[AGENTS_TRACE_HEADER_KEY];
  if (!payload) return null;
  return defaultPayloadConverter.fromPayload<AgentsSpanHeader>(payload);
}

/** Build an AgentsSpanHeader from the current OpenAI Agents trace/span context, or null if no trace is active. */
export function currentAgentsSpanHeader(): AgentsSpanHeader | null {
  const trace = getCurrentTrace();
  if (!trace) return null;
  const span = getCurrentSpan();
  return {
    traceName: trace.name ?? 'Unknown Workflow',
    spanId: span?.spanId ?? null,
    traceId: trace.traceId,
  };
}
