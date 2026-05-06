// Test workflows for OpenAI Agents SDK integration
// eslint-disable-next-line import/no-unassigned-import
import '@temporalio/openai-agents/lib/load-polyfills';

import {
  Agent,
  handoff,
  tool,
  addTraceProcessor,
  getCurrentTrace,
  withTrace,
  type ModelResponse,
} from '@openai/agents-core';
import { z } from 'zod';
import { webSearchTool } from '@openai/agents-openai';
import {
  ApplicationFailure,
  condition,
  defineSignal,
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
  isInWorkflow,
  isReplaying,
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

  const runner = new TemporalOpenAIRunner({ useLocalActivity: true, startToCloseTimeout: '60s' });
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
    startToCloseTimeout: '10s',
    retryPolicy: { maximumAttempts: 3, initialInterval: '100ms' },
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
 * otherwise the model call hits DummyModel and throws.
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
    summaryOverride: 'Custom model summary',
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
    startToCloseTimeout: '10s',
    retryPolicy: { maximumAttempts: 1 },
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
    startToCloseTimeout: '10s',
    retryPolicy: { maximumAttempts: 1 },
  });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}


/**
 * Workflow for testing that a plain Error without HTTP status/response
 * is classified as non-retryable.
 */
export async function plainErrorWorkflow(prompt: string): Promise<string> {
  const agent = new Agent({
    name: 'PlainErrorAgent',
    instructions: 'You are a helpful assistant.',
    model: 'gpt-4o-mini',
  });

  const runner = new TemporalOpenAIRunner({
    startToCloseTimeout: '10s',
    retryPolicy: { maximumAttempts: 3, initialInterval: '100ms' },
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
    priority: { priorityKey: 1 },
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
