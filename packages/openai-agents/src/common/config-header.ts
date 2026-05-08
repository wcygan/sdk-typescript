/**
 * Header-based propagation of plugin config from client → workflow isolate.
 *
 * Mirrors the encode/decode structure of `trace-header.ts`: encode/decode
 * helpers using `defaultPayloadConverter`, a well-known header key, and a
 * versioned wire shape. Adds a `__configVersion` field paralleling
 * {@link WIRE_VERSION} in `serialized-model.ts`.
 */
import { defaultPayloadConverter, type Headers } from '@temporalio/common';
import type { SerializableModelActivityOptions } from './model-activity-options';

/** Header key under which plugin config is propagated from client to workflow. */
export const AGENTS_CONFIG_HEADER_KEY = '__openai_agents_config';

/**
 * Wire format version. Bump when the shape changes incompatibly.
 * Parallels {@link WIRE_VERSION} in `serialized-model.ts` (model request wire protocol).
 */
export const CONFIG_WIRE_VERSION = 1;

/** Wire shape of the propagated plugin config. */
export interface AgentsConfigHeader {
  __configVersion: number;
  addTemporalSpans?: boolean;
  startTraces?: boolean;
  modelParams?: SerializableModelActivityOptions;
}

/**
 * Serialize an AgentsConfigHeader into a Temporal headers map.
 *
 * Always uses `defaultPayloadConverter` — never a user-configurable converter.
 * See `trace-header.ts` for the rationale: SDK-injected headers must use the
 * default converter so every cluster participant can decode them.
 */
export function injectAgentsConfigHeader(headers: Headers, config: AgentsConfigHeader): Headers {
  return { ...headers, [AGENTS_CONFIG_HEADER_KEY]: defaultPayloadConverter.toPayload(config) };
}

/**
 * Deserialize an AgentsConfigHeader from a Temporal headers map; returns null if absent
 * or if the version is unrecognized.
 */
export function extractAgentsConfigHeader(headers: Headers): AgentsConfigHeader | null {
  const payload = headers[AGENTS_CONFIG_HEADER_KEY];
  if (!payload) return null;
  const decoded = defaultPayloadConverter.fromPayload<AgentsConfigHeader>(payload);
  if (!decoded || decoded.__configVersion !== CONFIG_WIRE_VERSION) return null;
  return decoded;
}
