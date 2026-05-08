import type { AgentInputItem, ModelProvider, ModelRequest, ModelResponse } from '@openai/agents-core';
import { APIError } from 'openai';
import { ApplicationFailure } from '@temporalio/common';
import { heartbeat, activityInfo } from '@temporalio/activity';
import {
  type InvokeModelActivityInput,
  type JsonValue,
  type SerializedModelRequest,
  type SerializedModelResponse,
  WIRE_VERSION,
} from '../common/serialized-model';

/** Projects an upstream ModelResponse to its JSON-serializable wire form. */
export function toSerializedModelResponse(response: ModelResponse): SerializedModelResponse {
  return {
    __wireVersion: WIRE_VERSION,
    // Usage is a class whose add() method is stripped by JSON serialization.
    // All data properties are JSON-safe — projected field-by-field so each field's safety is
    // verifiable at the call site. requestUsageEntries contains RequestUsage class instances,
    // also projected explicitly.
    usage: {
      requests: response.usage.requests,
      inputTokens: response.usage.inputTokens,
      outputTokens: response.usage.outputTokens,
      totalTokens: response.usage.totalTokens,
      inputTokensDetails: response.usage.inputTokensDetails,
      outputTokensDetails: response.usage.outputTokensDetails,
      ...(response.usage.requestUsageEntries !== undefined && {
        requestUsageEntries: response.usage.requestUsageEntries.map((entry) => ({
          inputTokens: entry.inputTokens,
          outputTokens: entry.outputTokens,
          totalTokens: entry.totalTokens,
          inputTokensDetails: entry.inputTokensDetails,
          outputTokensDetails: entry.outputTokensDetails,
          endpoint: entry.endpoint,
        })),
      }),
    } as JsonValue,
    // AgentOutputItem[] variants are Zod-inferred plain objects — no class instances, no methods.
    // Double-cast required: optional `providerData?: Record<string, any>` allows `undefined` values
    // which prevents TS from narrowing the discriminated union to JsonValue in a single step.
    output: response.output as unknown as JsonValue[],
    responseId: response.responseId,
    providerData: response.providerData as Record<string, JsonValue> | undefined,
  };
}

function fromSerializedModelRequest(wire: SerializedModelRequest): ModelRequest {
  // Field-by-field construction — TS structurally validates the overall shape.
  // Only 3 fields need narrowing casts (widened to JsonValue during serialization).
  // __wireVersion deliberately stripped — internal protocol field, not part of upstream ModelRequest.
  return {
    systemInstructions: wire.systemInstructions,
    input: wire.input as string | AgentInputItem[],
    modelSettings: wire.modelSettings,
    tools: wire.tools,
    toolsExplicitlyProvided: wire.toolsExplicitlyProvided,
    outputType: wire.outputType,
    handoffs: wire.handoffs,
    // Indexed access: Prompt and ModelTracing types are not exported from @openai/agents-core.
    prompt: wire.prompt as ModelRequest['prompt'],
    previousResponseId: wire.previousResponseId,
    conversationId: wire.conversationId,
    tracing: wire.tracing as ModelRequest['tracing'],
    overridePromptModel: wire.overridePromptModel,
  };
}

/**
 * Creates the model activity functions to be registered with the Worker.
 * The returned activities use the provided ModelProvider to resolve models
 * and execute real LLM calls.
 */
export function createModelActivity(modelProvider: ModelProvider): {
  invokeModelActivity: (input: InvokeModelActivityInput) => Promise<SerializedModelResponse>;
} {
  return {
    async invokeModelActivity(input: InvokeModelActivityInput): Promise<SerializedModelResponse> {
      if (input.request.__wireVersion !== WIRE_VERSION) {
        throw ApplicationFailure.nonRetryable(
          `OpenAI Agents wire version mismatch: payload=${input.request.__wireVersion}, runtime=${WIRE_VERSION}. ` +
            `Upgrade workers and clients together.`,
          'WireVersionMismatch'
        );
      }
      // Shape validation beyond version check is intentionally minimal: no Zod or runtime schema
      // validation. The wire version literal + structural projection in toSerializedModelRequest
      // cover the actual risks (version skew and field leakage). Adding a runtime validator would
      // introduce a dependency with no concrete safety gain — upstream types are JSON-safe by design.

      const model = await Promise.resolve(modelProvider.getModel(input.modelName));

      const info = activityInfo();
      let heartbeatTimer: ReturnType<typeof setTimeout> | undefined;
      let stopped = false;
      if (info.heartbeatTimeoutMs && info.heartbeatTimeoutMs > 0) {
        const interval = info.heartbeatTimeoutMs / 2;
        const scheduleHeartbeat = () => {
          heartbeatTimer = setTimeout(() => {
            if (stopped) return;
            try {
              heartbeat();
            } catch {
              // Activity might be cancelled — ignore heartbeat errors
            }
            scheduleHeartbeat();
          }, interval);
        };
        scheduleHeartbeat();
      }

      try {
        const response = await model.getResponse(fromSerializedModelRequest(input.request));
        return toSerializedModelResponse(response);
      } catch (error) {
        if (error instanceof APIError) {
          const status = error.status;
          const headers = error.headers;

          // Prefer retry-after-ms (OpenAI-specific, millisecond precision) over standard Retry-After (seconds, converted to ms)
          let nextRetryDelay: number | undefined;
          if (headers) {
            const ms = headers.get('retry-after-ms');
            if (ms) {
              const parsed = parseFloat(ms);
              if (!Number.isNaN(parsed)) nextRetryDelay = parsed;
            }
            if (nextRetryDelay === undefined) {
              const s = headers.get('retry-after');
              if (s) {
                const parsed = parseFloat(s);
                if (!Number.isNaN(parsed)) nextRetryDelay = parsed * 1000;
              }
            }
          }

          // x-should-retry header overrides status-based classification
          let nonRetryable: boolean;
          const shouldRetry = headers?.get('x-should-retry');
          if (shouldRetry === 'true') {
            nonRetryable = false;
          } else if (shouldRetry === 'false') {
            nonRetryable = true;
          } else if (status !== undefined && (status === 408 || status === 409 || status === 429 || status >= 500)) {
            nonRetryable = false;
          } else {
            nonRetryable = true;
          }

          // Map status to error subtype
          let type: string;
          if (status === 429) type = 'ModelInvocationError.RateLimit';
          else if (status === 401 || status === 403) type = 'ModelInvocationError.Authentication';
          else if (status === 400 || status === 422) type = 'ModelInvocationError.BadRequest';
          else if (status === 408) type = 'ModelInvocationError.Timeout';
          else if (status === 409) type = 'ModelInvocationError.Conflict';
          else if (status !== undefined && status >= 500) type = 'ModelInvocationError.ServerError';
          else type = 'ModelInvocationError';

          throw ApplicationFailure.create({
            message: `Model invocation failed: ${error.message}`,
            type,
            nonRetryable,
            cause: error,
            ...(nextRetryDelay !== undefined ? { nextRetryDelay } : {}),
          });
        }

        // Non-APIError: wrap generically and let Temporal's retry policy decide
        const message = error instanceof Error ? error.message : String(error);
        throw ApplicationFailure.create({
          message: `Model invocation failed: ${message}`,
          type: 'ModelInvocationError',
          nonRetryable: false,
          cause: error instanceof Error ? error : new Error(String(error)),
        });
      } finally {
        stopped = true;
        if (heartbeatTimer) {
          clearTimeout(heartbeatTimer);
        }
      }
    },
  };
}
