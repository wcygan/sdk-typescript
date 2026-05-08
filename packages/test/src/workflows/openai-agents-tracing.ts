// eslint-disable-next-line import/no-unassigned-import
import '@temporalio/openai-agents/lib/load-polyfills';

import {
  Agent,
  getCurrentTrace,
  handoff,
  tool,
  setTracingDisabled,
  withTrace,
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
  makeContinueAsNewFunc,
  proxyActivities,
  proxyLocalActivities,
  setHandler,
  startChild,
  workflowInfo,
} from '@temporalio/workflow';
import {
  activityAsTool,
  TemporalOpenAIRunner,
  statelessMcpServer,
  getCurrentPluginConfig,
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

// --- Multi-workflow config isolation test workflow ---

const proceedSignal = defineSignal<[]>('proceed');

/**
 * Guards against per-workflow plugin config leaking between workflows that
 * share a V8 isolate. Before Item 21, plugin config was stored on globalThis,
 * which silently leaked between concurrent workflows under `reuseV8Context:
 * true`. With the per-workflow plugin-config-store, each workflow sees only
 * its own config.
 *
 * When `waitForSignal` is true, the workflow blocks on the `proceed` signal
 * after populating the store, enabling the test to verify that a concurrent
 * workflow B (with different config) does not overwrite A's config while A
 * is still alive.
 */
export async function configIsolationWorkflow(
  addTemporalSpans: boolean,
  waitForSignal?: boolean
): Promise<boolean> {
  // Constructing the runner populates the per-workflow store
  new TemporalOpenAIRunner({ addTemporalSpans });

  if (waitForSignal) {
    let proceed = false;
    setHandler(proceedSignal, () => {
      proceed = true;
    });
    await condition(() => proceed, '30 seconds');
  }

  // Read config AFTER the signal wait — if B's config leaked into A's store
  // slot, this would return B's value instead of A's.
  const config = getCurrentPluginConfig();
  return config?.addTemporalSpans ?? false;
}

// --- Signal trace propagation E2E test workflows (Item 26) ---

const signalTraceTestSignal = defineSignal('signalTraceTestSignal');
const childReadyQuery = defineQuery<boolean>('childReady');

/**
 * Child workflow for signal trace propagation E2E test. Waits for a signal
 * and captures the trace context visible in the signal handler.
 */
export async function signalTraceChildWorkflow(): Promise<string> {
  // Initialize tracing so temporal:handleSignal spans are emitted
  new TemporalOpenAIRunner({ addTemporalSpans: true });

  let ready = false;
  let signalTraceId = '';

  setHandler(childReadyQuery, () => ready);
  setHandler(signalTraceTestSignal, () => {
    signalTraceId = getCurrentTrace()?.traceId ?? 'NO_SIGNAL_TRACE';
  });

  ready = true;
  await condition(() => signalTraceId !== '', '30 seconds');
  return signalTraceId;
}

/**
 * Parent workflow for signal trace propagation E2E test. Starts a child
 * workflow, waits for it to be ready, then signals it within an agent trace
 * context. With addTemporalSpans=true, the outbound interceptor creates a
 * `temporal:signalWorkflow` span, and the child's inbound interceptor
 * creates a `temporal:handleSignal` span nested under the propagated trace.
 */
export async function signalTraceParentWorkflow(): Promise<{
  parentTraceId: string;
  signalTraceId: string;
}> {
  new TemporalOpenAIRunner({ addTemporalSpans: true });

  const handle = await startChild(signalTraceChildWorkflow);

  return withTrace('signal-e2e-test', async (trace) => {
    const parentTraceId = trace.traceId;
    await handle.signal(signalTraceTestSignal);
    const signalTraceId = await handle.result();
    return { parentTraceId, signalTraceId };
  });
}

// --- Idempotency flag test workflow (Item 33) ---

/**
 * Simple workflow that constructs a TemporalOpenAIRunner (triggering
 * ensureTracingProcessorRegistered) and runs a basic agent. Used by the
 * idempotency flag test to verify that running two workflows on the same
 * worker with reuseV8Context=true doesn't register the processor twice
 * (which would cause duplicate OTel spans).
 */
export async function idempotencyFlagWorkflow(prompt: string): Promise<string> {
  const runner = new TemporalOpenAIRunner({ addTemporalSpans: true });
  const agent = new Agent({
    name: 'IdempotencyTestAgent',
    instructions: 'Respond briefly.',
    model: 'gpt-4o-mini',
  });
  const result = await runner.run(agent, prompt);
  return result.finalOutput ?? '';
}

// --- ContinueAsNew trace context test workflow (Item 35) ---

/**
 * ContinueAsNew trace context test: first run establishes a trace context
 * via withTrace and calls continueAsNew. The second run reads getCurrentTrace()
 * and returns the traceId. If the outbound continueAsNew interceptor correctly
 * re-injects the __openai_span header, the continued execution's traceId
 * should match the original.
 */
export async function traceContinueAsNewWorkflow(iteration: number): Promise<{
  originalTraceId: string;
  continuedTraceId: string;
}> {
  new TemporalOpenAIRunner({ addTemporalSpans: true });

  if (iteration === 0) {
    return withTrace('continue-as-new-trace-test', async (trace) => {
      const originalTraceId = trace.traceId;
      const doContinueAsNew = makeContinueAsNewFunc<typeof traceContinueAsNewWorkflow>({
        memo: { originalTraceId },
      });
      await doContinueAsNew(1);
      // Never reached — continueAsNew throws
      return { originalTraceId: '', continuedTraceId: '' };
    });
  }

  // Second run — trace context should be restored from the continueAsNew header
  const continuedTrace = getCurrentTrace();
  const continuedTraceId = continuedTrace?.traceId ?? 'NO_TRACE';

  // Read the original trace ID from memo
  const memo = workflowInfo().memo;
  const originalTraceId = (memo?.originalTraceId as string) ?? 'NO_MEMO';

  return { originalTraceId, continuedTraceId };
}
