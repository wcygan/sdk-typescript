// Test workflows for OpenAI Agents SDK integration
// eslint-disable-next-line import/no-unassigned-import
import '@temporalio/openai-agents/lib/load-polyfills';

import {
  Agent,
  handoff,
  tool,
  addTraceProcessor,
  getCurrentTrace,
  setTracingDisabled,
  withTrace,
  type ModelResponse,
} from '@openai/agents-core';

// Tests opt back into agent-SDK tracing because upstream auto-disables it under NODE_ENV=test;
// the production plugin defers to upstream's default.
setTracingDisabled(false);
import { z } from 'zod';
import { webSearchTool } from '@openai/agents-openai';
import {
  ApplicationFailure,
  condition,
  continueAsNew,
  defineQuery,
  defineSignal,
  defineUpdate,
  executeChild,
  proxyActivities,
  setHandler,
  startChild,
  workflowInfo,
} from '@temporalio/workflow';
import {
  activityAsTool,
  TemporalOpenAIRunner,
  statelessMcpServer,
  statefulMcpServer,
  StatefulMCPServerReference,
  isInWorkflow,
  isReplaying,
  getCurrentPluginConfig,
  toSerializedModelRequest,
  type TemporalMCPServer,
} from '@temporalio/openai-agents/lib/workflow';
import type * as activities from '../activities/openai-agents';

/**
 * Basic workflow that creates an agent and runs it with a prompt.
 * The agent's model is automatically replaced with an ActivityBackedModel
 * by the runner, so LLM calls go through activities.
 */
export async function basicAgentWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'TestAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Workflow that uses an agent with a tool backed by a Temporal activity.
 * The getWeather tool is wrapped via activityAsTool(), so when the model
 * requests a tool call, it schedules the getWeather activity.
 */
export async function toolAgentWorkflow(prompt: string): Promise<string> {
  const weatherTool = activityAsTool<{ location: string }, Awaited<ReturnType<typeof activities.getWeather>>>({
    name: 'getWeather',
    description: 'Get the weather for a given city',
    parameters: {
      type: 'object',
      properties: {
        location: { type: 'string', description: 'The city name' },
      },
      required: ['location'],
      additionalProperties: false,
    },
    // Type reference only — not called in the workflow
    activityFn: null! as typeof activities.getWeather,
  });

  const agent = new Agent({
    name: 'WeatherAgent',
    instructions: 'You are a weather assistant. Use the getWeather tool when asked about weather.',
    model: 'gpt-4o-mini',
    tools: [weatherTool],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { maxTurns: 5 });
  return result.finalOutput ?? '';
}

/**
 * Workflow that tests agent handoffs. The TriageAgent hands off to the
 * WeatherSpecialist when it receives a weather-related question.
 */
export async function handoffAgentWorkflow(question: string): Promise<string> {
  const weatherSpecialist = new Agent({
    name: 'WeatherSpecialist',
    instructions: 'You are a weather specialist.',
    handoffDescription: 'Weather questions',
    model: 'fake-model',
  });

  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialists.',
    model: 'fake-model',
    handoffs: [weatherSpecialist],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(triageAgent, question, { maxTurns: 10 });
  return result.finalOutput ?? '';
}

/**
 * Workflow that tests the maxTurns option. Returns output and turn count.
 */
export async function maxTurnsAgentWorkflow(
  prompt: string,
  maxTurns: number
): Promise<{ output: string; turnCount: number }> {
  const agent = new Agent({
    name: 'TurnsAgent',
    instructions: 'You are a helpful assistant.',
    model: 'fake-model',
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { maxTurns });
  return { output: result.finalOutput ?? '', turnCount: result.rawResponses.length };
}

/**
 * Workflow with an agent that has multiple tools (getWeather + calculateSum).
 * Tests that multiple activity-backed tools work together.
 */
export async function multiToolAgentWorkflow(prompt: string): Promise<string> {
  const weatherTool = activityAsTool<{ location: string }, Awaited<ReturnType<typeof activities.getWeather>>>({
    name: 'getWeather',
    description: 'Get the weather for a given city',
    parameters: {
      type: 'object',
      properties: {
        location: { type: 'string', description: 'The city name' },
      },
      required: ['location'],
      additionalProperties: false,
    },
    activityFn: null! as typeof activities.getWeather,
  });

  const sumTool = activityAsTool<{ a: number; b: number }, Awaited<ReturnType<typeof activities.calculateSum>>>({
    name: 'calculateSum',
    description: 'Calculate the sum of two numbers',
    parameters: {
      type: 'object',
      properties: {
        a: { type: 'number', description: 'First number' },
        b: { type: 'number', description: 'Second number' },
      },
      required: ['a', 'b'],
      additionalProperties: false,
    },
    activityFn: null! as typeof activities.calculateSum,
  });

  const agent = new Agent({
    name: 'MultiToolAgent',
    instructions: 'Use tools to answer questions.',
    model: 'fake-model',
    tools: [weatherTool, sumTool],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { maxTurns: 10 });
  return result.finalOutput ?? '';
}

/**
 * Workflow that passes typed context through the runner.
 */
interface UserContext {
  userId: string;
  preferences: { language: string };
}

export async function contextAgentWorkflow(prompt: string, userId: string): Promise<string> {
  const agent = new Agent<UserContext>({
    name: 'ContextAgent',
    instructions: 'You are a helpful assistant.',
    model: 'fake-model',
  });

  const context: UserContext = {
    userId,
    preferences: { language: 'en' },
  };

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { context });
  return result.finalOutput ?? '';
}

/**
 * Workflow that passes a raw function as a tool instead of using activityAsTool().
 * convertAgent should reject this with a clear error.
 */
export async function rawFunctionToolWorkflow(question: string): Promise<string> {
  const rawFunction = async ({ location }: { location: string }) => {
    return { weather: 'sunny', location };
  };

  const agent = new Agent({
    name: 'RawToolAgent',
    instructions: 'Test agent with raw function tool.',
    tools: [rawFunction as any],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, question);
  return result.finalOutput ?? '';
}

/**
 * Workflow that passes runConfig.model as a string to override the agent's model.
 * The string model name should be wrapped with ActivityBackedModel by the runner.
 */
export async function runConfigStringModelWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'ModelOverrideAgent',
    instructions: 'You are a helpful assistant.',
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { runConfig: { model: 'gpt-4o-mini' } });
  return result.finalOutput ?? '';
}

/**
 * Workflow that uses local activities for model invocations.
 * The model call should appear as a local activity marker in the history.
 */
export async function localActivityAgentWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'LocalActivityAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner({ modelParams: { useLocalActivity: true, startToCloseTimeout: '60s' } });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Workflow with explicit retry policy for testing Temporal-level activity retries.
 * Uses maximumAttempts: 3 so if the model always throws a retryable error,
 * Temporal retries and then fails after exhausting attempts.
 */
export async function retryableModelWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'RetryAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner({
    modelParams: { startToCloseTimeout: '10s', retryPolicy: { maximumAttempts: 3, initialInterval: '100ms' } },
  });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Workflow where the agent's instructions function throws a plain Error.
 * This triggers the runner's catch block which wraps non-Temporal errors
 * as ApplicationFailure with type 'AgentsWorkflowError'.
 */
export async function agentsWorkflowErrorWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'ThrowingAgent',
    instructions: () => {
      throw new Error('Instructions evaluation failed');
    },
    model: 'fake-model',
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Workflow that uses an agent with a stateless MCP server.
 * The MCP server delegates listTools and callTool to Temporal activities.
 */
export async function mcpAgentWorkflow(prompt: string): Promise<string> {
  const mcpServer = statelessMcpServer('testMcp');

  const agent = new Agent({
    name: 'McpAgent',
    instructions: 'You have access to MCP tools.',
    model: 'gpt-4o-mini',
    mcpServers: [mcpServer],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { maxTurns: 5 });
  return result.finalOutput ?? '';
}

/**
 * Workflow that uses an agent with a built-in hosted tool (webSearchTool).
 * Verifies that hosted tools pass through without serialization error.
 */
export async function builtInToolAgentWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'SearchAgent',
    instructions: 'You have web search.',
    model: 'gpt-4o-mini',
    tools: [webSearchTool()],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

// --- Edge-case workflows ---

/**
 * Uses handoff(agent) wrapper (Handoff instance, not raw Agent in handoffs array).
 * The Handoff's inner agent must get its model replaced with ActivityBackedModel;
 * otherwise the model call hits PlaceholderModel and throws.
 */
export async function handoffInstanceWorkflow(question: string): Promise<string> {
  const weatherSpecialist = new Agent({
    name: 'WeatherSpecialist',
    instructions: 'You are a weather specialist.',
    handoffDescription: 'Weather questions',
    model: 'fake-model',
  });

  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialists.',
    model: 'fake-model',
    handoffs: [handoff(weatherSpecialist)],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(triageAgent, question, { maxTurns: 5 });
  return result.finalOutput ?? '';
}

/**
 * Two agents with cyclic handoff references (A → B → A).
 * Verifies convertAgent terminates without stack overflow.
 */
export async function cyclicHandoffWorkflow(prompt: string): Promise<string> {
  const agentA = new Agent({
    name: 'AgentA',
    instructions: 'You are agent A.',
    model: 'fake-model',
  });
  const agentB = new Agent({
    name: 'AgentB',
    instructions: 'You are agent B.',
    model: 'fake-model',
  });
  (agentA as any).handoffs = [agentB];
  (agentB as any).handoffs = [agentA];

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agentA, prompt);
  return result.finalOutput ?? '';
}

/**
 * Agent with a prompt template. The prompt field on ModelRequest must
 * survive serialization through ActivityBackedModel.
 */
export async function promptFieldWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'PromptAgent',
    instructions: 'You are a helpful assistant.',
    model: 'fake-model',
    prompt: {
      promptId: 'pt_test',
      variables: {},
    },
  } as any);

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Agent with a non-string model (object instead of string).
 * The runner must throw immediately with a clear error.
 */
export async function nonStringModelWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'BadModelAgent',
    instructions: 'You are a helpful assistant.',
    model: {} as any,
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Error whose .cause chain contains a TemporalFailure (ApplicationFailure).
 * Simulates agents-core wrapping a Temporal failure in its own exception.
 * The runner must walk .cause and re-throw the inner TemporalFailure.
 */
export async function wrappedTemporalFailureWorkflow(prompt: string): Promise<string> {
  const inner = ApplicationFailure.create({
    message: 'Inner temporal failure',
    type: 'InnerFailureType',
    nonRetryable: true,
  });
  const wrapper = new Error('Agents wrapper error');
  wrapper.cause = inner;

  const agent = new Agent({
    name: 'WrapperAgent',
    instructions: () => {
      throw wrapper;
    },
    model: 'fake-model',
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Catches the runner's error INSIDE the workflow to inspect the error shape.
 * The runner throws ApplicationFailure with the original Error as cause.
 */
export async function agentsWorkflowErrorClassCheckWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'ThrowingAgent',
    instructions: () => {
      throw new Error('Instructions evaluation failed');
    },
    model: 'fake-model',
  });

  const runner = new TemporalOpenAIRunner();
  try {
    await runner.run(agent, prompt);
    return 'no-error';
  } catch (e: any) {
    return JSON.stringify({
      errorName: e?.name ?? 'unknown',
      causeName: e?.cause?.name ?? 'none',
    });
  }
}

/**
 * Tests EventTarget polyfill listener error isolation.
 * The polyfill must wrap listeners in try/catch so one throwing listener
 * doesn't prevent subsequent listeners from firing.
 */
export async function eventTargetListenerErrorWorkflow(): Promise<{
  secondListenerCalled: boolean;
  dispatchSucceeded: boolean;
}> {
  const ET = (globalThis as any).EventTarget;
  const Evt = (globalThis as any).Event;
  const et = new ET();
  let secondCalled = false;
  et.addEventListener('test', () => {
    throw new Error('listener error');
  });
  et.addEventListener('test', () => {
    secondCalled = true;
  });
  let succeeded = false;
  try {
    et.dispatchEvent(new Evt('test'));
    succeeded = true;
  } catch {
    succeeded = false;
  }
  return { secondListenerCalled: secondCalled, dispatchSucceeded: succeeded };
}

/**
 * Tests EventTarget polyfill sets event.target and event.currentTarget.
 */
export async function eventTargetTargetFieldWorkflow(): Promise<{
  targetDefined: boolean;
  currentTargetDefined: boolean;
}> {
  const ET = (globalThis as any).EventTarget;
  const Evt = (globalThis as any).Event;
  const et = new ET();
  let targetVal: unknown;
  let currentTargetVal: unknown;
  et.addEventListener('test', (e: any) => {
    targetVal = e.target;
    currentTargetVal = e.currentTarget;
  });
  et.dispatchEvent(new Evt('test'));
  return {
    targetDefined: targetVal !== undefined && targetVal !== null,
    currentTargetDefined: currentTargetVal !== undefined && currentTargetVal !== null,
  };
}

/**
 * Tests Date field serialization in ModelResponse.
 * Temporal's JSON converter coerces Date objects to ISO strings.
 */
export async function dateInResponseWorkflow(prompt: string): Promise<{
  dateFieldType: string;
  hasDateField: boolean;
}> {
  const agent = new Agent({
    name: 'DateAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  const raw = result.rawResponses[0] as any;
  const dateField = raw?.createdAt;
  return {
    dateFieldType: dateField instanceof Date ? 'Date' : typeof dateField,
    hasDateField: dateField !== undefined,
  };
}

/**
 * Calls runner.runStreamed() which does not exist on TemporalOpenAIRunner.
 * Exercises the runtime path (via `as any`) to verify it fails cleanly.
 */
export async function runStreamedWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'StreamAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner();
  const stream = await (runner as any).runStreamed(agent, prompt);
  let output = '';
  for await (const event of stream) {
    if (event?.data?.text) output += event.data.text;
  }
  return output || '';
}

/**
 * Uses tool() from agents-core directly instead of activityAsTool().
 * Deterministic tool() products run inline in the workflow — no activity overhead.
 * The tool's execute callback must be deterministic (no I/O, no randomness).
 */
export async function directToolFactoryWorkflow(prompt: string): Promise<string> {
  const inlineTool = tool({
    name: 'inlineTool',
    description: 'A deterministic tool that runs inline in the workflow',
    parameters: {
      type: 'object' as const,
      properties: {
        input: { type: 'string' },
      },
      required: ['input'] as const,
      additionalProperties: false as const,
    },
    execute: async (_ctx, args) => {
      return `processed: ${(args as any).input}`;
    },
  });

  const agent = new Agent({
    name: 'DirectToolAgent',
    instructions: 'Test agent with direct tool() factory tool.',
    model: 'fake-model',
    tools: [inlineTool],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Workflow that tests MCP listPrompts and getPrompt via activities.
 */
export async function mcpPromptsWorkflow(_prompt: string): Promise<{
  prompts: unknown[];
  promptResult: unknown;
}> {
  const mcpServer = statelessMcpServer('testMcp') as TemporalMCPServer;

  const prompts = await mcpServer.listPrompts();
  const promptResult = await mcpServer.getPrompt('greeting', { name: 'World' });

  return { prompts, promptResult };
}

/**
 * Workflow that tests MCP factoryArgument passthrough.
 */
export async function mcpFactoryArgWorkflow(prompt: string): Promise<string> {
  const mcpServer = statelessMcpServer('testMcp', {
    factoryArgument: { tenantId: 'tenant-42' },
  });

  const agent = new Agent({
    name: 'McpFactoryArgAgent',
    instructions: 'You have access to MCP tools.',
    model: 'gpt-4o-mini',
    mcpServers: [mcpServer],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { maxTurns: 5 });
  return result.finalOutput ?? '';
}

/**
 * Workflow that uses MCP server via StatelessMCPServerProvider-registered activities.
 */
export async function mcpProviderWorkflow(prompt: string): Promise<string> {
  const mcpServer = statelessMcpServer('providerMcp');

  const agent = new Agent({
    name: 'McpProviderAgent',
    instructions: 'You have access to MCP tools.',
    model: 'gpt-4o-mini',
    mcpServers: [mcpServer],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { maxTurns: 5 });
  return result.finalOutput ?? '';
}

/**
 * Workflow that uses summaryOverride string in model params.
 */
export async function summaryOverrideStringWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'SummaryAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner({
    modelParams: { summaryOverride: 'Custom model summary' },
  });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Verifies tracing utilities return expected values from workflow context.
 */
export async function tracingUtilitiesWorkflow(): Promise<{
  isInWf: boolean;
  isReplay: boolean;
}> {
  return {
    isInWf: isInWorkflow(),
    isReplay: isReplaying(),
  };
}

/**
 * Agent with explicit model 'original-model'. The test overrides via runConfig.model
 * to 'override-model' and verifies the activity receives the override.
 */
export async function runConfigModelOverrideCheckWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'OverrideCheckAgent',
    instructions: 'You are a helpful assistant.',
    model: 'original-model',
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt, { runConfig: { model: 'override-model' } });
  return result.finalOutput ?? '';
}

/**
 * Handoff agent has a raw function tool that should be rejected.
 * convertAgent must recurse into handoff targets to validate their tools.
 */
export async function handoffWithRawToolWorkflow(prompt: string): Promise<string> {
  const specialist = new Agent({
    name: 'SpecialistWithRawTool',
    instructions: 'You are a specialist.',
    model: 'fake-model',
    tools: [(() => 'raw result') as any],
  });

  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialists.',
    model: 'fake-model',
    handoffs: [specialist],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(triageAgent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Same as handoffWithRawToolWorkflow but using handoff() wrapper instance.
 */
export async function handoffInstanceWithRawToolWorkflow(prompt: string): Promise<string> {
  const specialist = new Agent({
    name: 'SpecialistWithRawTool',
    instructions: 'You are a specialist.',
    model: 'fake-model',
    tools: [(() => 'raw result') as any],
  });

  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialists.',
    model: 'fake-model',
    handoffs: [handoff(specialist)],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(triageAgent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Tests that convertAgent does not mutate the original Handoff object.
 * The original handoff's agent must still have its original model after run.
 */
export async function handoffMutationCheckWorkflow(prompt: string): Promise<string> {
  const specialist = new Agent({
    name: 'Specialist',
    instructions: 'You are a specialist.',
    model: 'specialist-model',
  });

  const handoffObj = handoff(specialist);
  const originalAgentModel = typeof (handoffObj.agent as any).model;

  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialists.',
    model: 'fake-model',
    handoffs: [handoffObj],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(triageAgent, prompt);

  const afterAgentModel = typeof (handoffObj.agent as any).model;

  return JSON.stringify({
    output: result.finalOutput ?? '',
    originalModelType: originalAgentModel,
    afterModelType: afterAgentModel,
    mutated: originalAgentModel !== afterAgentModel,
  });
}

/**
 * Workflow with handoff that has onHandoff callback.
 * convertAgent must preserve onInvokeHandoff so the callback fires.
 */
export async function handoffOnHandoffCallbackWorkflow(prompt: string): Promise<{
  output: string;
  onHandoffCalled: boolean;
}> {
  let onHandoffCalled = false;

  const specialist = new Agent({
    name: 'CallbackSpecialist',
    instructions: 'You are a specialist.',
    model: 'fake-model',
  });

  const handoffObj = handoff(specialist, {
    onHandoff: async (_ctx: any, _input?: { reason: string }) => {
      onHandoffCalled = true;
    },
    inputType: z.object({
      reason: z.string().describe('Reason for handoff'),
    }),
  });

  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialists.',
    model: 'fake-model',
    handoffs: [handoffObj],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(triageAgent, prompt, { maxTurns: 5 });

  return {
    output: result.finalOutput ?? '',
    onHandoffCalled,
  };
}

/**
 * Handoff with isEnabled=false. If convertAgent drops isEnabled,
 * the handoff defaults to always-enabled and appears in the model's tool list.
 */
export async function handoffIsEnabledFalseWorkflow(prompt: string): Promise<string> {
  const specialist = new Agent({
    name: 'DisabledSpecialist',
    instructions: 'You are a specialist.',
    model: 'fake-model',
  });

  const handoffObj = handoff(specialist, {
    isEnabled: false,
  });

  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialists.',
    model: 'fake-model',
    handoffs: [handoffObj],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(triageAgent, prompt, { maxTurns: 3 });
  return result.finalOutput ?? '';
}

/**
 * Handoff with custom inputJsonSchema. convertAgent must preserve the schema.
 */
export async function handoffWithCustomSchemaWorkflow(prompt: string): Promise<string> {
  const specialist = new Agent({
    name: 'SchemaSpecialist',
    instructions: 'You are a specialist.',
    model: 'fake-model',
  });

  const handoffObj = handoff(specialist);
  (handoffObj as any).inputJsonSchema = {
    type: 'object',
    properties: {
      reason: { type: 'string', description: 'Reason for handoff' },
    },
    required: ['reason'],
    additionalProperties: false,
  };
  (handoffObj as any).strictJsonSchema = true;

  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialists.',
    model: 'fake-model',
    handoffs: [handoffObj],
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(triageAgent, prompt, { maxTurns: 3 });
  return result.finalOutput ?? '';
}

/**
 * Workflow for testing 408 Timeout error classification.
 */
export async function timeoutErrorWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'TimeoutAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner({
    modelParams: { startToCloseTimeout: '10s', retryPolicy: { maximumAttempts: 1 } },
  });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Workflow for testing x-should-retry header override.
 */
export async function xShouldRetryWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'XShouldRetryAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner({
    modelParams: { startToCloseTimeout: '10s', retryPolicy: { maximumAttempts: 1 } },
  });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}


/**
 * Workflow for testing that a plain Error without HTTP status/response
 * is retryable (defers to Temporal's retry policy).
 */
export async function plainErrorWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'PlainErrorAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner({
    modelParams: { startToCloseTimeout: '10s', retryPolicy: { maximumAttempts: 3, initialInterval: '100ms' } },
  });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Workflow that uses priority in model params.
 */
export async function extendedModelParamsWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'ExtendedParamsAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner({
    modelParams: { priority: { priorityKey: 1 } },
  });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

// --- Wire contract tests ---

/**
 * Wire round-trip: verifies that populated Prompt-shaped and ModelTracing-shaped
 * objects survive workflow→activity→workflow through the wire contract.
 * Returns both the final output and response-side metadata for assertion.
 */
export async function wireRoundTripWorkflow(prompt: string): Promise<{
  finalOutput: string;
  usageInputTokens: number;
  usageOutputTokens: number;
  outputLength: number;
  hasWireVersion: boolean;
}> {
  const agent = new Agent({
    name: 'WireRoundTripAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
    prompt: {
      promptId: 'pt_round_trip',
      variables: { key: 'value', nested: { deep: true } },
    },
  } as any);

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  const raw: ModelResponse | undefined = result.rawResponses[0];
  return {
    finalOutput: result.finalOutput ?? '',
    usageInputTokens: raw?.usage?.inputTokens ?? -1,
    usageOutputTokens: raw?.usage?.outputTokens ?? -1,
    outputLength: Array.isArray(raw?.output) ? raw.output.length : -1,
    hasWireVersion: '__wireVersion' in (raw ?? {}),
  };
}

/**
 * Wire stripping: verifies that signal is not present in the
 * request received by the activity-side model.
 */
export async function wireStrippingCheckWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'WireStrippingAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner();
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

/**
 * Wire version mismatch: directly proxies the invokeModelActivity activity and
 * calls it with __wireVersion: 999 to trigger the version check.
 *
 * True cross-version testing would require running two package versions simultaneously,
 * which is impractical. This test verifies the activity-side guard responds correctly
 * to a stale payload.
 */
export async function wireVersionMismatchWorkflow(): Promise<{ errorType: string; errorMessage: string }> {
  const activities = proxyActivities<{
    invokeModelActivity(input: unknown): Promise<unknown>;
  }>({ startToCloseTimeout: '30s' });

  try {
    await activities.invokeModelActivity({
      modelName: 'test-model',
      request: {
        __wireVersion: 999,
        input: 'test',
        modelSettings: {},
        tools: [],
        outputType: { type: 'text' },
        handoffs: [],
      },
    });
    return { errorType: 'none', errorMessage: 'no error' };
  } catch (e: any) {
    return {
      errorType: e?.cause?.type ?? 'unknown',
      errorMessage: e?.cause?.message ?? String(e),
    };
  }
}

// --- Wire shape snapshot workflow ---

export async function wireRequestSnapshotWorkflow(): Promise<string[]> {
  const request = {
    systemInstructions: 'test instructions',
    input: 'test input',
    modelSettings: { temperature: 0.5 },
    tools: [],
    toolsExplicitlyProvided: true,
    outputType: { type: 'text' },
    handoffs: [],
    prompt: { promptId: 'pt_test' },
    previousResponseId: 'resp_123',
    conversationId: 'conv_456',
    tracing: false,
    overridePromptModel: true,
    signal: 'should-be-stripped',
  } as any;

  const wire = toSerializedModelRequest(request);
  return Object.keys(wire).sort();
}

/**
 * Verifies that the OpenAI Agents SDK tracing path is active and that
 * TemporalTracingProcessor receives trace/span events during an agent run.
 */
export async function tracingSpanCaptureWorkflow(): Promise<{
  traceIds: string[];
  spanTypes: string[];
}> {
  const capture: { traceIds: string[]; spanTypes: string[] } = { traceIds: [], spanTypes: [] };

  const runner = new TemporalOpenAIRunner();

  addTraceProcessor({
    async onTraceStart(trace: any) {
      capture.traceIds.push(trace.traceId);
    },
    async onTraceEnd() {},
    async onSpanStart(span: any) {
      capture.spanTypes.push(span.spanData.type);
    },
    async onSpanEnd() {},
    async shutdown() {},
    async forceFlush() {},
  });

  const agent = new Agent({
    name: 'TracingTestAgent',
    instructions: 'You are a test agent.',
    model: 'fake-model',
  });

  await runner.run(agent, 'Hello');

  return capture;
}

/**
 * Replay-safety test workflow. Designed to run with maxCachedWorkflows: 0
 * so the worker evicts the workflow after each task and replays from scratch.
 */
export async function replaySafetyWorkflow(): Promise<{
  traceIds: string[];
  spanTypes: string[];
  replayDetected: boolean;
}> {
  const capture: { traceIds: string[]; spanTypes: string[]; replayDetected: boolean } = {
    traceIds: [],
    spanTypes: [],
    replayDetected: isReplaying(),
  };

  const runner = new TemporalOpenAIRunner();

  addTraceProcessor({
    async onTraceStart(trace: any) {
      capture.traceIds.push(trace.traceId);
    },
    async onTraceEnd() {},
    async onSpanStart(span: any) {
      capture.spanTypes.push(span.spanData.type);
    },
    async onSpanEnd() {},
    async shutdown() {},
    async forceFlush() {},
  });

  const agent = new Agent({
    name: 'ReplayTestAgent',
    instructions: 'You are a test agent.',
    model: 'fake-model',
  });

  await runner.run(agent, 'Hello');

  return capture;
}

export async function handoffCloneSnapshotWorkflow(): Promise<{
  fieldsPreserved: Record<string, boolean>;
  agentReplaced: boolean;
  onInvokeHandoffReplaced: boolean;
  prototypeMatch: boolean;
}> {
  const specialist = new Agent({
    name: 'SnapshotSpecialist',
    instructions: 'You are a specialist.',
    model: 'snapshot-model',
  });

  const inputFilterFn = (data: any) => data;

  const handoffObj = handoff(specialist, {
    toolNameOverride: 'custom_snapshot_tool',
    toolDescriptionOverride: 'Custom snapshot description',
    onHandoff: async () => {},
    inputType: z.object({ reason: z.string() }),
    inputFilter: inputFilterFn,
    isEnabled: false,
  });

  // Clone using the exact same Object.create technique as convertAgent
  const clone = Object.create(
    Object.getPrototypeOf(handoffObj),
    Object.getOwnPropertyDescriptors(handoffObj)
  ) as typeof handoffObj;

  // Simulate what convertAgent does: replace agent and onInvokeHandoff
  const replacementAgent = new Agent({
    name: 'ReplacementSpecialist',
    instructions: 'Replacement.',
    model: 'replacement-model',
  });
  clone.agent = replacementAgent;
  const replacementOnInvoke = async () => replacementAgent;
  clone.onInvokeHandoff = replacementOnInvoke;

  return {
    fieldsPreserved: {
      toolName: clone.toolName === handoffObj.toolName,
      toolDescription: clone.toolDescription === handoffObj.toolDescription,
      inputJsonSchema: JSON.stringify(clone.inputJsonSchema) === JSON.stringify(handoffObj.inputJsonSchema),
      strictJsonSchema: clone.strictJsonSchema === handoffObj.strictJsonSchema,
      agentName: clone.agentName === handoffObj.agentName,
      inputFilter: clone.inputFilter === handoffObj.inputFilter,
      isEnabled: clone.isEnabled === handoffObj.isEnabled,
    },
    agentReplaced: clone.agent !== handoffObj.agent,
    onInvokeHandoffReplaced: clone.onInvokeHandoff !== handoffObj.onInvokeHandoff,
    prototypeMatch: Object.getPrototypeOf(clone) === Object.getPrototypeOf(handoffObj),
  };
}

export async function concurrentTracingIsolationWorkflow(): Promise<{
  traceIds: string[];
  spanTypes: string[];
  workflowId: string;
}> {
  const capture: { traceIds: string[]; spanTypes: string[]; workflowId: string } = {
    traceIds: [],
    spanTypes: [],
    workflowId: workflowInfo().workflowId,
  };

  const runner = new TemporalOpenAIRunner();

  addTraceProcessor({
    async onTraceStart(trace: any) {
      capture.traceIds.push(trace.traceId);
    },
    async onTraceEnd() {},
    async onSpanStart(span: any) {
      capture.spanTypes.push(span.spanData.type);
    },
    async onSpanEnd() {},
    async shutdown() {},
    async forceFlush() {},
  });

  const agent = new Agent({
    name: 'IsolationTestAgent',
    instructions: 'You are a test agent for isolation testing.',
    model: 'fake-model',
  });

  await runner.run(agent, 'Hello');

  return capture;
}

export async function traceContextPropagationWorkflow(): Promise<{
  workflowTraceId: string;
  activityTraceId: string;
}> {
  let workflowTraceId = '';

  const runner = new TemporalOpenAIRunner();

  addTraceProcessor({
    async onTraceStart(trace: any) {
      workflowTraceId = trace.traceId;
    },
    async onTraceEnd() {},
    async onSpanStart() {},
    async onSpanEnd() {},
    async shutdown() {},
    async forceFlush() {},
  });

  const agent = new Agent({
    name: 'TracePropagationAgent',
    instructions: 'You are a test agent.',
    model: 'trace-capture-model',
  });

  const result = await runner.run(agent, 'Hello');
  const activityTraceId = result.finalOutput?.replace('TRACE:', '') ?? '';

  return { workflowTraceId, activityTraceId };
}

export async function clientToWorkflowTraceWorkflow(): Promise<string> {
  new TemporalOpenAIRunner();
  return getCurrentTrace()?.traceId ?? 'NO_TRACE';
}

const traceTestSignal = defineSignal('traceTestSignal');
const queryTraceTestDone = defineSignal('queryTraceTestDone');

export async function signalTracePropagationParentWorkflow(): Promise<{
  parentTraceId: string;
  signalTraceId: string;
}> {
  new TemporalOpenAIRunner();

  return withTrace('signal-trace-test', async (trace) => {
    const parentTraceId = trace.traceId;
    const handle = await startChild(signalTracePropagationChildWorkflow);
    await handle.signal(traceTestSignal);
    const signalTraceId = await handle.result();
    return { parentTraceId, signalTraceId };
  });
}

export async function signalTracePropagationChildWorkflow(): Promise<string> {
  let capturedTraceId = '';
  setHandler(traceTestSignal, () => {
    capturedTraceId = getCurrentTrace()?.traceId ?? 'NO_SIGNAL_TRACE';
  });
  await condition(() => capturedTraceId !== '', '10 seconds');
  return capturedTraceId;
}

export async function childWorkflowTracePropagationParentWorkflow(): Promise<{
  parentTraceId: string;
  childTraceId: string;
}> {
  new TemporalOpenAIRunner();

  return withTrace('child-trace-test', async (trace) => {
    const parentTraceId = trace.traceId;
    const childTraceId = await executeChild(childWorkflowTracePropagationChildWorkflow);
    return { parentTraceId, childTraceId };
  });
}

export async function childWorkflowTracePropagationChildWorkflow(): Promise<string> {
  return getCurrentTrace()?.traceId ?? 'NO_TRACE';
}

export async function deterministicTraceIdsWorkflow(): Promise<{
  traceIds: string[];
  spanIds: string[];
  spanStartTimestamps: string[];
  workflowTimestamp: string;
}> {
  const traceIds: string[] = [];
  const spanIds: string[] = [];
  const spanStartTimestamps: string[] = [];

  // Capture Date-based timestamp — the V8 sandbox replaces Date with a
  // deterministic clock. If this weren't deterministic, replay would fail
  // with NondeterminismError because the command sequence would diverge.
  const workflowTimestamp = new Date().toISOString();

  const runner = new TemporalOpenAIRunner();

  addTraceProcessor({
    async onTraceStart(trace: any) {
      traceIds.push(trace.traceId);
    },
    async onTraceEnd() {},
    async onSpanStart(span: any) {
      spanIds.push(span.spanId);
      if (span.startedAt) spanStartTimestamps.push(span.startedAt);
    },
    async onSpanEnd() {},
    async shutdown() {},
    async forceFlush() {},
  });

  const agent = new Agent({
    name: 'IdTestAgent',
    instructions: 'Respond briefly.',
    model: 'gpt-4o-mini',
  });

  await runner.run(agent, 'Hi');
  return { traceIds, spanIds, spanStartTimestamps, workflowTimestamp };
}

/**
 * Workflow that uses a stateful MCP server with a short scheduleToStartTimeout.
 * When no dedicated worker is running, tool calls will time out and produce
 * a DedicatedWorkerFailure ApplicationFailure.
 */
export async function statefulMcpNoWorkerWorkflow(scheduleToStartTimeoutMs: number): Promise<string> {
  const server = statefulMcpServer('testStateful', {
    config: {
      startToCloseTimeout: '1 minute',
      scheduleToStartTimeout: `${scheduleToStartTimeoutMs} milliseconds`,
    },
  });

  await server.connect();
  try {
    await server.listTools();
    return 'unexpected-success';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) {
      return `${err.type}: ${err.message}`;
    }
    throw err;
  } finally {
    await server.cleanup();
  }
}

/**
 * Workflow that creates a stateful MCP server, connects, calls tools, and cleans up.
 */
export async function statefulMcpAgentWorkflow(prompt: string): Promise<string> {
  const server = statefulMcpServer('testStateful');

  await server.connect();
  try {
    const agent = new Agent({
      name: 'StatefulMcpAgent',
      instructions: 'You have access to MCP tools.',
      model: 'gpt-4o-mini',
      mcpServers: [server],
    });

    const runner = new TemporalOpenAIRunner();
    const result = await runner.run(agent, prompt, { maxTurns: 5 });
    return result.finalOutput ?? '';
  } finally {
    await server.cleanup();
  }
}

/**
 * Workflow that tests calling listTools before connect throws an error.
 */
export async function statefulMcpNotConnectedWorkflow(): Promise<string> {
  const server = statefulMcpServer('testStateful');
  try {
    await server.listTools();
    return 'unexpected-success';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) {
      return err.message;
    }
    throw err;
  }
}

/**
 * Workflow for testing multi-run isolation under reuseV8Context.
 * Connects to a stateful MCP server, calls a tool, and returns the result.
 * Each run's server instance is keyed by runId, so two runs on the same
 * V8 isolate must see their own server's data.
 */
export async function statefulMcpIsolationWorkflow(): Promise<string> {
  const server = statefulMcpServer('isolationTest');

  await server.connect();
  try {
    const tools = await server.listTools();
    if (tools.length === 0) return 'no-tools';
    const result = await server.callTool(tools[0].name, null);
    return JSON.stringify(result);
  } finally {
    await server.cleanup();
  }
}

/**
 * Workflow for testing heartbeat-timeout failure on operation activities.
 * Sets a tight heartbeatTimeout on the operation config so that a blocking
 * operation activity (listTools that hangs) triggers a heartbeat timeout,
 * which handleWorkerFailure maps to DedicatedWorkerFailure.
 *
 * Uses a long scheduleToStartTimeout to ensure the dedicated worker picks
 * up the activity before that timeout fires — the heartbeat timeout must
 * be the one that fires first.
 */
export async function statefulMcpHeartbeatTimeoutWorkflow(): Promise<string> {
  const server = statefulMcpServer('heartbeatTest', {
    config: {
      startToCloseTimeout: '30 seconds',
      heartbeatTimeout: '1 second',
      retryPolicy: { maximumAttempts: 1 },
    },
  });

  await server.connect();
  try {
    await server.listTools();
    return 'unexpected-success';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) {
      return `${err.type}: ${err.message}`;
    }
    throw err;
  } finally {
    await server.cleanup();
  }
}

/**
 * Workflow for testing that the session activity heartbeats during a slow
 * server.connect(). Uses serverSessionConfig.heartbeatTimeout so the
 * timeout applies to the long-running session activity itself, not the
 * operation activities.
 *
 * The server's connect() blocks for ~1.5s. With a 3s heartbeatTimeout on
 * the session config, the heartbeat mechanism must fire during connect()
 * to prevent the timeout. After connect() completes, the workflow calls
 * listTools and cleans up normally.
 */
export async function statefulMcpSlowConnectHeartbeatWorkflow(): Promise<string> {
  const server = statefulMcpServer('slowConnectTest', {
    serverSessionConfig: {
      heartbeatTimeout: '3 seconds',
      startToCloseTimeout: '30 seconds',
    },
    config: {
      startToCloseTimeout: '10 seconds',
    },
  });

  await server.connect();
  try {
    const tools = await server.listTools();
    return `connected:${tools.length}`;
  } finally {
    await server.cleanup();
  }
}

/**
 * Workflow for replay-safety testing with maxCachedWorkflows: 0.
 * Connects to a stateful MCP server, calls a tool, cleans up, and returns.
 * The per-run task queue name is deterministic (based on runId), so replay
 * must produce the same command sequence without NondeterminismError.
 */
export async function statefulMcpReplayWorkflow(): Promise<string> {
  const server = statefulMcpServer('replayTest');

  await server.connect();
  try {
    const tools = await server.listTools();
    if (tools.length === 0) return 'no-tools';
    const result = await server.callTool(tools[0].name, {});
    return JSON.stringify(result);
  } finally {
    await server.cleanup();
  }
}

/**
 * Triggers ensureTracingProcessorRegistered (and its ALS context shape
 * smoke check) by constructing a TemporalOpenAIRunner, then returns
 * 'ok' if no error was thrown.
 */
export async function alsContextShapeSmokeCheckWorkflow(): Promise<string> {
  new TemporalOpenAIRunner();
  return 'ok';
}

/**
 * Returns the traceId from getCurrentTrace() at workflow start, before any
 * agent operations. Used to detect ALS context leaks between workflows.
 */
export async function alsLeakDetectionWorkflow(): Promise<string | null> {
  const trace = getCurrentTrace();
  return trace?.traceId ?? null;
}

// --- Query trace context propagation ---

const queryTraceIdQuery = defineQuery<string>('queryTraceId');

export async function queryTracePropagationWorkflow(): Promise<string> {
  new TemporalOpenAIRunner();

  setHandler(queryTraceIdQuery, () => {
    return getCurrentTrace()?.traceId ?? 'NO_QUERY_TRACE';
  });

  let done = false;
  setHandler(queryTraceTestDone, () => {
    done = true;
  });

  await condition(() => done, '30 seconds');
  return 'done';
}

// --- Config header propagation test workflows ---

const configUpdateWithStartUpdate = defineUpdate<
  { addTemporalSpans: boolean; taskQueue: string | undefined },
  []
>('configUpdateWithStart');

/**
 * Reads the per-workflow plugin config and returns addTemporalSpans and
 * modelParams.taskQueue. Used by header propagation tests to verify the
 * config header was decoded and stored correctly.
 */
export async function configPropagationWorkflow(): Promise<{
  addTemporalSpans: boolean;
  taskQueue: string | undefined;
}> {
  // Runner constructor populates the store from the header (if present)
  // or from its own args.
  new TemporalOpenAIRunner();
  const config = getCurrentPluginConfig();
  return {
    addTemporalSpans: config?.addTemporalSpans ?? false,
    taskQueue: config?.modelParams?.taskQueue,
  };
}

/**
 * Variant of configPropagationWorkflow with an update handler, used by the
 * startUpdateWithStart propagation test. The update returns the observed config;
 * the workflow waits for the update to fire before completing.
 */
export async function configUpdateWithStartWorkflow(): Promise<{
  addTemporalSpans: boolean;
  taskQueue: string | undefined;
}> {
  new TemporalOpenAIRunner();
  const config = getCurrentPluginConfig();
  const result = {
    addTemporalSpans: config?.addTemporalSpans ?? false,
    taskQueue: config?.modelParams?.taskQueue,
  };

  let updateFired = false;
  setHandler(configUpdateWithStartUpdate, () => {
    updateFired = true;
    return result;
  });

  await condition(() => updateFired, '30 seconds');
  return result;
}

/**
 * ContinueAsNew test: first run stores config and continues-as-new.
 * Second run (iteration=1) reads config from the store (propagated via
 * the outbound interceptor's continueAsNew header injection) and returns it.
 */
export async function configContinueAsNewWorkflow(iteration: number): Promise<{
  addTemporalSpans: boolean;
  taskQueue: string | undefined;
}> {
  new TemporalOpenAIRunner();

  if (iteration === 0) {
    await continueAsNew<typeof configContinueAsNewWorkflow>(1);
  }

  // Second run — config should have been propagated via continueAsNew header
  const config = getCurrentPluginConfig();
  return {
    addTemporalSpans: config?.addTemporalSpans ?? false,
    taskQueue: config?.modelParams?.taskQueue,
  };
}

/**
 * Child workflow that reads config from the store. Used by the parent
 * workflow to verify child workflows receive the config header.
 */
export async function configChildWorkflow(): Promise<{
  addTemporalSpans: boolean;
  taskQueue: string | undefined;
}> {
  new TemporalOpenAIRunner();
  const config = getCurrentPluginConfig();
  return {
    addTemporalSpans: config?.addTemporalSpans ?? false,
    taskQueue: config?.modelParams?.taskQueue,
  };
}

/**
 * Parent workflow that starts a child workflow and returns its config observation.
 * Tests that the outbound interceptor re-injects the config header for child workflows.
 */
export async function configChildParentWorkflow(): Promise<{
  addTemporalSpans: boolean;
  taskQueue: string | undefined;
}> {
  new TemporalOpenAIRunner();
  return executeChild(configChildWorkflow, { args: [] });
}

/**
 * Override precedence test: the runner constructor args should win over the
 * plugin config header values.
 */
export async function configOverridePrecedenceWorkflow(): Promise<{
  addTemporalSpans: boolean;
  taskQueue: string | undefined;
}> {
  // Runner overrides: addTemporalSpans=false (overriding plugin's true),
  // taskQueue='runner-override' (overriding plugin's header value)
  new TemporalOpenAIRunner({
    addTemporalSpans: false,
    modelParams: { taskQueue: 'runner-override' },
  });
  const config = getCurrentPluginConfig();
  return {
    addTemporalSpans: config?.addTemporalSpans ?? true,
    taskQueue: config?.modelParams?.taskQueue,
  };
}

/**
 * Fallback test: workflow started without the plugin client interceptor (no
 * config header). The runner constructor populates the store from its own args.
 */
export async function configFallbackWorkflow(): Promise<{
  addTemporalSpans: boolean;
}> {
  new TemporalOpenAIRunner({ addTemporalSpans: true });
  const config = getCurrentPluginConfig();
  return {
    addTemporalSpans: config?.addTemporalSpans ?? false,
  };
}

/**
 * H3 regression test: constructing multiple runners in the same workflow should
 * NOT accumulate modelParams from earlier runners. Each runner merges its own
 * args with the original header config, not the effective config left by the
 * previous runner.
 */
export async function configModelParamsIsolationWorkflow(): Promise<{
  r1TaskQueue: string | undefined;
  r2TaskQueue: string | undefined;
  r2StartToCloseTimeout: string | number | undefined;
}> {
  // Runner 1: sets taskQueue='a'
  new TemporalOpenAIRunner({ modelParams: { taskQueue: 'a' } });
  const config1 = getCurrentPluginConfig();
  const r1TaskQueue = config1?.modelParams?.taskQueue;

  // Runner 2: sets startToCloseTimeout='5s', does NOT set taskQueue.
  // If the bug exists, r2 would inherit taskQueue='a' from r1's effective config.
  new TemporalOpenAIRunner({ modelParams: { startToCloseTimeout: '5s' } });
  const config2 = getCurrentPluginConfig();
  const r2TaskQueue = config2?.modelParams?.taskQueue;
  const r2StartToCloseTimeout = config2?.modelParams?.startToCloseTimeout;

  return { r1TaskQueue, r2TaskQueue, r2StartToCloseTimeout };
}

/**
 * M1 regression test — child workflow for summaryOverride function-form stripping.
 * Reads summaryOverride AND taskQueue from the plugin config store and returns both.
 *
 * The parent sets a function-form summaryOverride AND a sibling `taskQueue` field
 * via the runner. The outbound interceptor's `injectConfigHeaderFromStore` must:
 * - Strip summaryOverride (non-string → omitted from wire header)
 * - Preserve taskQueue (serializable string → survives wire header)
 *
 * The sibling assertion makes the test rock-solid: even if JSON serialization
 * also happens to drop summaryOverride as a side-effect, the taskQueue surviving
 * proves the explicit M1 narrowing logic (destructure + selective reattach) is
 * doing meaningful work and preserving siblings correctly.
 */
export async function summaryOverrideFunctionStripChildWorkflow(): Promise<{
  summaryOverride: string | undefined;
  taskQueue: string | undefined;
}> {
  new TemporalOpenAIRunner();
  const config = getCurrentPluginConfig();
  return {
    summaryOverride: config?.modelParams?.summaryOverride as string | undefined,
    taskQueue: config?.modelParams?.taskQueue,
  };
}

/**
 * M1 regression test — parent workflow that sets a function-form summaryOverride
 * AND a serializable sibling field (`taskQueue`) via the runner constructor.
 *
 * The child reads both summaryOverride and taskQueue from the propagated config.
 *
 * **Discrimination design:**
 * - Pre-M1, `injectConfigHeaderFromStore` forwarded `config.modelParams` directly.
 *   JSON serialization would silently mangle the function form (drop the function
 *   method, leaving `{}` for the parent object or stripping the field entirely
 *   depending on JS spec corner cases).
 * - Post-M1, the function explicitly narrows summaryOverride (drops non-string
 *   values via destructure + selective reattach) BEFORE injecting into the wire
 *   header.
 * - The test discriminates by checking BOTH: summaryOverride is undefined (stripped)
 *   AND taskQueue survives (sibling preserved). The sibling proves the strip is
 *   targeted — not collateral damage from a broken serialization path.
 */
export async function summaryOverrideFunctionStripParentWorkflow(): Promise<{
  summaryOverride: string | undefined;
  taskQueue: string | undefined;
}> {
  new TemporalOpenAIRunner({
    modelParams: {
      summaryOverride: { provide: () => 'parent-summary' },
      taskQueue: 'sibling-task-queue',
    },
  });
  return executeChild(summaryOverrideFunctionStripChildWorkflow, { args: [] });
}
