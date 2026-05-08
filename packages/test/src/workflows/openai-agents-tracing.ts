// eslint-disable-next-line import/no-unassigned-import
import '@temporalio/openai-agents/lib/load-polyfills';

import {
  Agent,
  handoff,
  tool,
  setTracingDisabled,
  type InputGuardrail,
  type OutputGuardrail,
  type TextOutput,
} from '@openai/agents-core';

// Tests opt back into agent-SDK tracing because upstream auto-disables it under NODE_ENV=test;
// the production plugin defers to upstream's default.
setTracingDisabled(false);
import {
  condition,
  defineQuery,
  defineSignal,
  defineUpdate,
  executeChild,
  proxyActivities,
  proxyLocalActivities,
  setHandler,
} from '@temporalio/workflow';
import {
  activityAsTool,
  TemporalOpenAIRunner,
  statelessMcpServer,
} from '@temporalio/openai-agents/lib/workflow';
import type * as activities from '../activities/openai-agents';

// --- Queries, signals, and updates ---

const isWaitingForSignalQuery = defineQuery<boolean>('isWaitingForSignal');
const resumeSignal = defineSignal<[string]>('resume');
const validatedUpdate = defineUpdate<string, [string]>('validatedUpdate');
const unvalidatedUpdate = defineUpdate<string, [string]>('unvalidatedUpdate');

// --- Activities proxy ---

const acts = proxyActivities<typeof activities>({
  startToCloseTimeout: '10 seconds',
});

// --- Guardrails ---

const inputGuardrail: InputGuardrail = {
  name: 'profanity-check',
  execute: async ({ input }) => ({
    tripwireTriggered: typeof input === 'string' && input.includes('BLOCKED'),
    outputInfo: { checked: true },
  }),
};

const outputGuardrail: OutputGuardrail<TextOutput> = {
  name: 'length-check',
  execute: async ({ agentOutput }) => ({
    tripwireTriggered: typeof agentOutput === 'string' && agentOutput.length > 10000,
    outputInfo: { checked: true },
  }),
};

// --- Child workflows ---

export async function tracingChildWorkflow(prompt: string): Promise<string> {
  const childAgent = new Agent({
    name: 'ChildAgent',
    instructions: 'You are a child workflow agent.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner({ addTemporalSpans: true });
  const result = await runner.run(childAgent, prompt);
  return result.finalOutput ?? '';
}

export async function tracingChildWorkflowNoSpans(prompt: string): Promise<string> {
  const childAgent = new Agent({
    name: 'ChildAgent',
    instructions: 'You are a child workflow agent.',
    model: 'gpt-4o-mini',
  });
  const runner = new TemporalOpenAIRunner({ addTemporalSpans: false });
  const result = await runner.run(childAgent, prompt);
  return result.finalOutput ?? '';
}

// --- Core workflow body ---

async function runComprehensive(prompt: string, addTemporalSpans: boolean): Promise<string> {
  let waitingForSignal = false;
  let signalValue = '';
  let updateResult = '';

  setHandler(isWaitingForSignalQuery, () => waitingForSignal);

  setHandler(resumeSignal, (value: string) => {
    signalValue = value;
  });

  setHandler(validatedUpdate, (value: string) => {
    updateResult = value;
    return `validated:${value}`;
  }, {
    validator: (value: string) => {
      if (!value) throw new Error('empty update value');
    },
  });

  setHandler(unvalidatedUpdate, (value: string) => {
    return `unvalidated:${value}`;
  });

  const runner = new TemporalOpenAIRunner({ addTemporalSpans });

  // Regular activity called directly (outside agent context — no temporal:* spans)
  await acts.getWeather({ location: 'Tokyo' });

  // Local activity proxy — used both standalone and as an agent tool
  const localActs = proxyLocalActivities<typeof activities>({
    startToCloseTimeout: '10 seconds',
  });

  // Basic agent run (no tools)
  const basicAgent = new Agent({
    name: 'BasicAgent',
    instructions: 'Respond briefly.',
    model: 'gpt-4o-mini',
  });
  await runner.run(basicAgent, prompt);

  // Agent with local-activity-backed tool — exercises the
  // `temporal:startLocalActivity:*` wrapping path inside agent context.
  const calculateSumTool = tool({
    name: 'calculateSum',
    description: 'Adds two numbers',
    parameters: {
      type: 'object' as const,
      properties: { a: { type: 'number' }, b: { type: 'number' } },
      required: ['a', 'b'] as const,
      additionalProperties: false as const,
    },
    execute: async (_ctx, args) => {
      const result = await localActs.calculateSum({ a: (args as any).a, b: (args as any).b });
      return result;
    },
  });
  const localActivityAgent = new Agent({
    name: 'LocalActivityAgent',
    instructions: 'Use calculateSum tool.',
    model: 'gpt-4o-mini',
    tools: [calculateSumTool],
  });
  await runner.run(localActivityAgent, 'add 1 and 2', { maxTurns: 5 });

  // Agent with inline tool() factory
  const echoTool = tool({
    name: 'echo',
    description: 'Echoes input',
    parameters: {
      type: 'object' as const,
      properties: { text: { type: 'string' } },
      required: ['text'] as const,
      additionalProperties: false as const,
    },
    execute: async (_ctx, args) => `echo:${(args as any).text}`,
  });

  const toolAgent = new Agent({
    name: 'EchoToolAgent',
    instructions: 'Use the echo tool.',
    model: 'gpt-4o-mini',
    tools: [echoTool],
  });
  await runner.run(toolAgent, 'echo hello', { maxTurns: 5 });

  // Agent with activityAsTool
  const weatherTool = activityAsTool<{ location: string }, string>({
    name: 'getWeather',
    description: 'Get weather',
    parameters: {
      type: 'object',
      properties: { location: { type: 'string' } },
      required: ['location'],
      additionalProperties: false,
    },
    // activityAsTool only inspects the function name; the runtime call goes through the proxy.
    activityFn: null! as typeof activities.getWeather,
  });

  const activityToolAgent = new Agent({
    name: 'WeatherToolAgent',
    instructions: 'Use getWeather tool.',
    model: 'gpt-4o-mini',
    tools: [weatherTool],
  });
  await runner.run(activityToolAgent, 'weather in Tokyo', { maxTurns: 5 });

  // Multi-turn agent (3+ turns)
  const multiTurnAgent = new Agent({
    name: 'MultiTurnAgent',
    instructions: 'Check weather in multiple cities.',
    model: 'gpt-4o-mini',
    tools: [weatherTool],
  });
  await runner.run(multiTurnAgent, 'weather in Tokyo and London', { maxTurns: 10 });

  // Handoff
  const specialist = new Agent({
    name: 'Specialist',
    instructions: 'You are a specialist.',
    handoffDescription: 'Specialist agent',
    model: 'gpt-4o-mini',
  });
  const triageAgent = new Agent({
    name: 'TriageAgent',
    instructions: 'Route to specialist.',
    model: 'gpt-4o-mini',
    handoffs: [specialist],
  });
  await runner.run(triageAgent, 'route me', { maxTurns: 10 });

  // Stateless MCP server tool call
  const mcpServer = statelessMcpServer('test-mcp');
  const mcpAgent = new Agent({
    name: 'McpAgent',
    instructions: 'Use MCP tools.',
    model: 'gpt-4o-mini',
    mcpServers: [mcpServer],
  });
  await runner.run(mcpAgent, 'list tools', { maxTurns: 5 });

  // Input and output guardrails
  const guardedAgent = new Agent({
    name: 'GuardedAgent',
    instructions: 'Respond briefly.',
    model: 'gpt-4o-mini',
    inputGuardrails: [inputGuardrail],
    outputGuardrails: [outputGuardrail],
  });
  await runner.run(guardedAgent, 'safe input');

  // Child workflow with sub-agent
  const childWorkflow = addTemporalSpans ? tracingChildWorkflow : tracingChildWorkflowNoSpans;
  const childResult = await executeChild(childWorkflow, {
    args: ['child prompt'],
  });

  // Signal handler — wait for external signal
  waitingForSignal = true;
  await condition(() => signalValue !== '', '30 seconds');

  // Post-signal activity
  await acts.getWeather({ location: 'London' });

  // Wait for update
  await condition(() => updateResult !== '', '30 seconds');

  return `done:${childResult}:${signalValue}:${updateResult}`;
}

// --- Exported workflow functions ---

export async function comprehensiveAgentWorkflow(prompt: string): Promise<string> {
  return runComprehensive(prompt, true);
}

export async function comprehensiveAgentWorkflowNoSpans(prompt: string): Promise<string> {
  return runComprehensive(prompt, false);
}
