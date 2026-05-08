import type { Model, ModelProvider, ModelRequest, ModelResponse, StreamEvent } from '@openai/agents-core';

/**
 * A Model that throws if called. Used as a safety net — all model resolution
 * should go through ActivityBackedModel, so PlaceholderModel should never be invoked.
 */
class PlaceholderModel implements Model {
  async getResponse(_request: ModelRequest): Promise<ModelResponse> {
    throw new Error(
      'Temporal workflows must invoke OpenAI Agents models via activities. ' +
        'Use `TemporalOpenAIRunner` from `@temporalio/openai-agents` to run agents ' +
        'inside a workflow instead of invoking the agent directly.'
    );
  }

  // eslint-disable-next-line require-yield
  async *getStreamedResponse(_request: ModelRequest): AsyncIterable<StreamEvent> {
    throw new Error(
      'Temporal workflows must invoke OpenAI Agents models via activities. ' +
        'Use `TemporalOpenAIRunner` from `@temporalio/openai-agents` to run agents ' +
        'inside a workflow instead of invoking the agent directly.'
    );
  }
}

/**
 * Inert ModelProvider used to satisfy the SDK Runner's required `modelProvider`
 * parameter without letting the SDK construct its default `OpenAIProvider`.
 * The default would be benign at runtime (its `getModel` is never called once
 * `convertAgent` has replaced agent models with `ActivityBackedModel` instances),
 * but constructing it pulls OpenAI client setup into the workflow's call path.
 * This stub avoids that entirely. Should never be invoked — if it is, an agent
 * escaped `convertAgent`'s traversal.
 */
export class PlaceholderModelProvider implements ModelProvider {
  getModel(_modelName?: string): Model {
    return new PlaceholderModel();
  }
}
