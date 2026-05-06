import { defaultPayloadConverter, type Headers } from '@temporalio/common';

/** Header key under which agent trace context is propagated across Temporal boundaries. */
export const AGENTS_TRACE_HEADER_KEY = '__openai_span';

/** Wire shape of the propagated agent trace context. */
export interface AgentsSpanHeader {
  traceName: string;
  spanId: string | null;
  traceId: string | null;
}

/** Serialize an AgentsSpanHeader into a Temporal headers map (uses defaultPayloadConverter). */
export function injectAgentsTraceHeader(headers: Headers, info: AgentsSpanHeader): Headers {
  return { ...headers, [AGENTS_TRACE_HEADER_KEY]: defaultPayloadConverter.toPayload(info) };
}

/** Deserialize an AgentsSpanHeader from a Temporal headers map; returns null if absent. */
export function extractAgentsTraceHeader(headers: Headers): AgentsSpanHeader | null {
  const payload = headers[AGENTS_TRACE_HEADER_KEY];
  if (!payload) return null;
  return defaultPayloadConverter.fromPayload<AgentsSpanHeader>(payload);
}
