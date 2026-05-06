/**
 * Shared OTel bridge helpers for mapping OpenAI Agents SDK span data to
 * OpenTelemetry span names and attributes. Used by both the workflow-side
 * TemporalTracingProcessor and the activity-side ActivityTracingProcessor.
 */
import type * as otel from '@opentelemetry/api';
import type { SpanData } from '@openai/agents-core';

export const TRACER_NAME = '@temporalio/openai-agents';

/**
 * Converts an OpenAI Agents SDK trace ID (`trace_<32hex>`) to a valid
 * OTel trace ID (32 lowercase hex chars, 128 bits).
 *
 * The agent SDK trace ID embeds a UUID v4 hex string after the `trace_`
 * prefix, which is exactly 32 hex characters — a 1:1 fit for OTel's
 * 128-bit trace ID format. Both the workflow-side and activity-side
 * processors call this on the same agent trace ID to produce the same
 * OTel trace ID, unifying all spans into one trace tree.
 */
export function agentTraceIdToOtelTraceId(agentTraceId: string): string {
  const hex = agentTraceId.startsWith('trace_') ? agentTraceId.slice(6) : agentTraceId.replace(/-/g, '');
  return hex.padStart(32, '0').slice(0, 32).toLowerCase();
}

/**
 * Converts an OpenAI Agents SDK span ID (`span_<24hex>`) to a valid
 * OTel span ID (16 lowercase hex chars, 64 bits).
 *
 * The agent SDK span ID embeds 24 hex characters after the `span_`
 * prefix. OTel span IDs are 16 hex characters (64 bits), so we take
 * the first 16. Both sides apply the same truncation, producing
 * identical OTel span IDs from identical agent span IDs.
 */
export function agentSpanIdToOtelSpanId(agentSpanId: string): string {
  const hex = agentSpanId.startsWith('span_') ? agentSpanId.slice(5) : agentSpanId.replace(/-/g, '');
  return hex.padStart(16, '0').slice(0, 16).toLowerCase();
}

/**
 * OTel ID generator that can be seeded with specific trace/span IDs.
 * When seeded, the next `generateTraceId()` or `generateSpanId()` call
 * returns the seeded value (FIFO). When the seed queue is empty, falls
 * through to a simple random hex generator.
 *
 * Seeding is synchronous and consumed synchronously by `tracer.startSpan()`,
 * so there is no risk of async interleaving between seed and consumption
 * within a single processor call site.
 */
export class SeedableIdGenerator {
  private traceSeeds: string[] = [];
  private spanSeeds: string[] = [];

  seedTraceId(id: string): void {
    this.traceSeeds.push(id);
  }

  seedSpanId(id: string): void {
    this.spanSeeds.push(id);
  }

  generateTraceId(): string {
    if (this.traceSeeds.length > 0) return this.traceSeeds.shift()!;
    return randomHex(32);
  }

  generateSpanId(): string {
    if (this.spanSeeds.length > 0) return this.spanSeeds.shift()!;
    return randomHex(16);
  }
}

function randomHex(len: number): string {
  const bytes = new Uint8Array(len / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Replaces the `_idGenerator` on an OTel Tracer with a {@link SeedableIdGenerator}.
 * Returns the installed generator so callers can seed it before `startSpan()`.
 *
 * The OTel SDK's `Tracer` stores its ID generator as an internal `_idGenerator`
 * field (not part of the public API). This access is intentional: replacing
 * the TracerProvider's ID generator with a seedable variant is the only way
 * to make `tracer.startSpan()` produce deterministic IDs in the workflow sandbox.
 */
export function installSeedableIdGenerator(tracer: unknown): SeedableIdGenerator {
  const gen = new SeedableIdGenerator();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (tracer as any)._idGenerator = gen;
  return gen;
}

export function spanNameFromData(data: SpanData): string {
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
      if (data.name.startsWith('temporal:')) return data.name;
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

export function staticAttributesFromSpanData(data: SpanData): otel.Attributes {
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

export function dynamicAttributesFromSpanData(data: SpanData): otel.Attributes {
  const attrs: otel.Attributes = {};
  switch (data.type) {
    case 'agent':
      if (data.tools) attrs['openai.agents.agent.tools'] = data.tools.join(',');
      break;
    case 'generation':
      if (data.model) attrs['openai.agents.generation.model'] = data.model;
      break;
    case 'handoff':
      if (data.from_agent) attrs['openai.agents.handoff.from_agent'] = data.from_agent;
      if (data.to_agent) attrs['openai.agents.handoff.to_agent'] = data.to_agent;
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
