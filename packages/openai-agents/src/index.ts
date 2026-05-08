/**
 * @temporalio/openai-agents — Temporal integration for the OpenAI Agents SDK.
 *
 * Deferred (not in this package):
 * - nexusOperationAsTool — TS SDK lacks executeNexusOperation; add when available
 * - testing.AgentEnvironment — richer test harness beyond FakeModel
 * - workflowFailureExceptionTypes registration (TS SDK doesn't support)
 */

export { OpenAIAgentsPlugin } from './worker/plugin';
export type { OpenAIAgentsPluginOptions, MCPServerProvider } from './worker/plugin';
export { toSerializedModelResponse } from './worker/activities';
export { StatelessMCPServerProvider } from './worker/mcp-provider';
export type { StatelessMCPServerFactory, MCPToolDefinition, MCPCallToolResult } from './worker/mcp-provider';
export { StatefulMCPServerProvider } from './worker/stateful-mcp-provider';
export type { StatefulMCPServer } from './worker/stateful-mcp-provider';
export {
  WIRE_VERSION,
  type SerializedModelRequest,
  type SerializedModelResponse,
  type InvokeModelActivityInput,
  type JsonValue,
} from './common/serialized-model';
export type { ModelActivityOptions, ModelSummaryProvider, AgentInputItem } from './common/model-activity-options';
export { DEFAULT_MODEL_ACTIVITY_OPTIONS } from './common/model-activity-options';
export { ToolSerializationError } from './workflow/tools';
export type { ActivityToolDefinition, ActivityAsToolOptions, JsonObjectSchema } from './workflow/tools';
export type { StatelessMcpServerOptions, TemporalMCPServer, MCPPromptDefinition } from './workflow/mcp-client';
export {
  DEDICATED_WORKER_FAILURE_TYPE,
  DEDICATED_WORKER_SCHEDULE_FAILURE_MESSAGE,
  DEDICATED_WORKER_HEARTBEAT_FAILURE_MESSAGE,
} from './workflow/stateful-mcp-client';
export type { StatefulMcpServerOptions } from './workflow/stateful-mcp-client';
export {
  isInWorkflow,
  isReplaying,
  TemporalTracingProcessor,
  ensureTracingProcessorRegistered,
} from './workflow/tracing';
export type { TemporalTracingProcessorOptions } from './workflow/tracing';
export type { TemporalOpenAIRunnerOptions } from './workflow/runner';
export type { OpenAIAgentsTraceInterceptorOptions } from './worker/trace-interceptor';
export { OpenAIAgentsTraceClientInterceptor } from './client/trace-interceptor';
export type { OpenAIAgentsTraceClientInterceptorOptions } from './client/trace-interceptor';
export { AGENTS_TRACE_HEADER_KEY, type AgentsSpanHeader } from './common/trace-header';

export * as testing from './worker/testing';
