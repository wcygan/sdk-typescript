import { setTracingDisabled } from '@openai/agents-core';
import * as otelApi from '@opentelemetry/api';
import { ExportResultCode } from '@opentelemetry/core';
import { BasicTracerProvider, SimpleSpanProcessor, type ReadableSpan } from '@opentelemetry/sdk-trace-base';
import * as opentelemetry from '@opentelemetry/sdk-node';
import { SEMRESATTRS_SERVICE_NAME } from '@opentelemetry/semantic-conventions';
import test from 'ava';
import { v4 as uuid4 } from 'uuid';
import { OpenTelemetryPlugin } from '@temporalio/interceptors-opentelemetry';
import {
  OpenAIAgentsPlugin,
  StatelessMCPServerProvider,
  createTracerProvider,
  ReplaySafeTracerProvider,
  TemporalIdGenerator,
} from '@temporalio/openai-agents';
import { DefaultLogger, Runtime, bundleWorkflowCode } from '@temporalio/worker';
import {
  comprehensiveAgentWorkflow,
  comprehensiveAgentWorkflowNoSpans,
  tracingChildWorkflow,
  tracingChildWorkflowNoSpans,
  configIsolationWorkflow,
  signalTraceParentWorkflow,
  signalTraceChildWorkflow,
  idempotencyFlagWorkflow,
  traceContinueAsNewWorkflow,
} from './workflows/openai-agents-tracing';
import { FakeModelProvider, textResponse, toolCallResponse, handoffResponse } from './stubs/openai-agents';
import * as agentActivities from './activities/openai-agents';
import { bundlerOptions, RUN_INTEGRATION_TESTS, Worker } from './helpers';
import { createTestWorkflowEnvironment } from './helpers-integration';

// Tests opt back into agent-SDK tracing because upstream auto-disables it under NODE_ENV=test;
// the production plugin defers to upstream's default.
setTracingDisabled(false);

// --- In-memory span collector + tree builder ---

/**
 * Builds indented span hierarchies grouped by OTel trace ID.
 * Children are sorted by startTime for deterministic output.
 * Returns one string[] per root span — each entry is an indented line.
 */
function buildSpanTree(spans: ReadableSpan[]): string[][] {
  const byTraceId = new Map<string, ReadableSpan[]>();
  for (const s of spans) {
    const tid = s.spanContext().traceId;
    const arr = byTraceId.get(tid) ?? [];
    arr.push(s);
    byTraceId.set(tid, arr);
  }

  const traces: string[][] = [];
  for (const [, traceSpans] of byTraceId) {
    const children = new Map<string | undefined, ReadableSpan[]>();
    for (const s of traceSpans) {
      const pid = s.parentSpanId;
      const arr = children.get(pid) ?? [];
      arr.push(s);
      children.set(pid, arr);
    }

    // Sort children by startTime, then by name for deterministic output
    for (const [, kids] of children) {
      kids.sort((a, b) => {
        const [aS, aNs] = a.startTime;
        const [bS, bNs] = b.startTime;
        if (aS !== bS) return aS - bS;
        if (aNs !== bNs) return aNs - bNs;
        return a.name.localeCompare(b.name);
      });
    }

    const roots = children.get(undefined) ?? [];
    for (const root of roots) {
      const tree: string[] = [];
      const walk = (spanId: string, depth: number) => {
        const kids = children.get(spanId) ?? [];
        for (const kid of kids) {
          tree.push('  '.repeat(depth) + kid.name);
          walk(kid.spanContext().spanId, depth + 1);
        }
      };
      tree.push(root.name);
      walk(root.spanContext().spanId, 1);
      traces.push(tree);
    }
  }
  return traces;
}

/**
 * Extracts agent SDK (`openai.agents.*`) and temporal interceptor
 * (`temporal:*`) spans from all OTel traces, preserving parent-child
 * nesting. Agent spans live in their own OTel traces (one per
 * `runner.run()` call) with deterministic trace IDs derived from agent
 * SDK trace IDs. Non-matching spans are transparent — their children
 * get promoted. Traces are sorted by first span start time so the
 * output order matches workflow execution order.
 *
 * @param spans - All collected OTel spans
 * @returns Indented hierarchy of agent+temporal spans across all traces
 */
function extractAgentHierarchy(spans: ReadableSpan[]): string[] {
  // External-event handler spans (query/signal/update) nest under whatever workflow span
  // is active at the yield point — exclude them since their position is timing-dependent.
  const isExternalEventSpan = (name: string) =>
    name.startsWith('temporal:handleQuery') ||
    name.startsWith('temporal:handleSignal') ||
    name.startsWith('temporal:handleUpdate') ||
    name.startsWith('temporal:validateUpdate');
  const isAgentSpan = (name: string) =>
    (name.startsWith('openai.agents.') || name.startsWith('temporal:')) && !isExternalEventSpan(name);

  // Group spans by OTel trace ID
  const byTraceId = new Map<string, ReadableSpan[]>();
  for (const s of spans) {
    const tid = s.spanContext().traceId;
    const arr = byTraceId.get(tid) ?? [];
    arr.push(s);
    byTraceId.set(tid, arr);
  }

  // Filter to traces that contain at least one agent/temporal span
  const agentTraces: { traceId: string; spans: ReadableSpan[]; minTime: [number, number] }[] = [];
  for (const [traceId, traceSpans] of byTraceId) {
    if (traceSpans.some((s) => isAgentSpan(s.name))) {
      const minTime = traceSpans.reduce(
        (min, s) => {
          const [aS, aNs] = s.startTime;
          const [mS, mNs] = min;
          if (aS < mS || (aS === mS && aNs < mNs)) return s.startTime;
          return min;
        },
        [Infinity, 0] as [number, number]
      );
      agentTraces.push({ traceId, spans: traceSpans, minTime });
    }
  }

  // Sort traces by earliest span start time
  agentTraces.sort((a, b) => {
    const [aS, aNs] = a.minTime;
    const [bS, bNs] = b.minTime;
    if (aS !== bS) return aS - bS;
    return aNs - bNs;
  });

  const result: string[] = [];

  for (const { spans: traceSpans } of agentTraces) {
    // Build adjacency: parentSpanId → children
    const children = new Map<string | undefined, ReadableSpan[]>();
    for (const s of traceSpans) {
      const pid = s.parentSpanId;
      const arr = children.get(pid) ?? [];
      arr.push(s);
      children.set(pid, arr);
    }

    // Sort children by startTime then name
    for (const [, kids] of children) {
      kids.sort((a, b) => {
        const [aS, aNs] = a.startTime;
        const [bS, bNs] = b.startTime;
        if (aS !== bS) return aS - bS;
        if (aNs !== bNs) return aNs - bNs;
        return a.name.localeCompare(b.name);
      });
    }

    // Walk the tree. Agent/temporal spans are emitted at `depth`; others
    // are transparent (children promoted to parent level).
    const walk = (parentId: string | undefined, depth: number) => {
      const kids = children.get(parentId) ?? [];
      for (const kid of kids) {
        if (isAgentSpan(kid.name)) {
          result.push('  '.repeat(depth) + kid.name);
          walk(kid.spanContext().spanId, depth + 1);
        } else {
          walk(kid.spanContext().spanId, depth);
        }
      }
    };

    walk(undefined, 0);
  }

  return result;
}

// --- MCP provider ---

function createMcpProvider(): StatelessMCPServerProvider {
  return new StatelessMCPServerProvider('test-mcp', {
    async listTools() {
      return [
        {
          name: 'lookup',
          description: 'Look up info',
          inputSchema: {
            type: 'object' as const,
            properties: { query: { type: 'string' } },
            required: ['query'],
            additionalProperties: false,
          },
        },
      ];
    },
    async callTool(arg) {
      return [{ type: 'text', text: `result for ${(arg.args as any)?.query ?? 'unknown'}` }];
    },
    async listPrompts() {
      return [];
    },
    async getPrompt() {
      return {};
    },
  });
}

// --- Shared OTel + plugin setup ---

function createOtelContext(): {
  spans: ReadableSpan[];
  provider: ReplaySafeTracerProvider;
  otelPlugin: OpenTelemetryPlugin;
} {
  const spans: ReadableSpan[] = [];
  const resource = new opentelemetry.resources.Resource({ [SEMRESATTRS_SERVICE_NAME]: 'test-tracing' });
  const provider = createTracerProvider({ resource });
  provider.addSpanProcessor(
    new SimpleSpanProcessor({
      export(exportedSpans, resultCallback) {
        spans.push(...exportedSpans);
        resultCallback({ code: ExportResultCode.SUCCESS });
      },
      async shutdown() {},
    })
  );
  otelApi.trace.setGlobalTracerProvider(provider);

  const otelPlugin = new OpenTelemetryPlugin({
    resource,
    spanProcessor: new SimpleSpanProcessor({
      export(exportedSpans, resultCallback) {
        spans.push(...exportedSpans);
        resultCallback({ code: ExportResultCode.SUCCESS });
      },
      async shutdown() {},
    }),
  });

  return { spans, provider, otelPlugin };
}

// --- Model response sequences ---

function* comprehensiveResponses() {
  // BasicAgent: single text
  yield textResponse('basic-done');
  // LocalActivityAgent: tool call then text
  yield toolCallResponse('calculateSum', { a: 1, b: 2 });
  yield textResponse('local-activity-done');
  // EchoToolAgent: tool call then text
  yield toolCallResponse('echo', { text: 'hello' });
  yield textResponse('echo-done');
  // WeatherToolAgent: tool call then text
  yield toolCallResponse('getWeather', { location: 'Tokyo' });
  yield textResponse('weather-done');
  // MultiTurnAgent: two tool calls then text
  yield toolCallResponse('getWeather', { location: 'Tokyo' });
  yield toolCallResponse('getWeather', { location: 'London' });
  yield textResponse('multi-turn-done');
  // TriageAgent→Specialist handoff: handoff then text
  yield handoffResponse('transfer_to_Specialist');
  yield textResponse('specialist-done');
  // McpAgent: tool call to MCP tool then text
  yield toolCallResponse('lookup', { query: 'test' });
  yield textResponse('mcp-done');
  // GuardedAgent: single text
  yield textResponse('guarded-done');
  // ChildAgent (child workflow): single text
  yield textResponse('child-done');
}

// --- Expected agent trace hierarchies ---
//
// These literals capture the exact agent SDK + temporal interceptor span
// tree from the workflow's OTel trace. OTel plugin spans (StartActivity,
// RunWorkflow, etc.) are filtered out to avoid sensitivity to their
// nondeterministic ordering. Any change to agent span nesting, temporal
// span placement, or agent loop order breaks the assertion.
//
// Activity-side `temporal:executeActivity` spans live in separate OTel
// traces and are asserted separately via name presence.

// Each runner.run() call produces its own OTel trace with a deterministic
// trace ID derived from the agent SDK trace ID. Activity-side
// `temporal:executeActivity` spans share the same trace ID and nest under
// `temporal:startActivity:*` via derived parent span IDs.
//
const EXPECTED_AGENT_HIERARCHY_WITH_TEMPORAL_SPANS = [
  'openai.agents.run',
  '  openai.agents.agent:BasicAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  'openai.agents.run',
  '  openai.agents.agent:LocalActivityAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  '    openai.agents.function:calculateSum',
  '      temporal:startLocalActivity:calculateSum',
  '        temporal:executeActivity',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  'openai.agents.run',
  '  openai.agents.agent:EchoToolAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  '    openai.agents.function:echo',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  'openai.agents.run',
  '  openai.agents.agent:WeatherToolAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  '    openai.agents.function:getWeather',
  '      temporal:startActivity:getWeather',
  '        temporal:executeActivity',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  'openai.agents.run',
  '  openai.agents.agent:MultiTurnAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  '    openai.agents.function:getWeather',
  '      temporal:startActivity:getWeather',
  '        temporal:executeActivity',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  '    openai.agents.function:getWeather',
  '      temporal:startActivity:getWeather',
  '        temporal:executeActivity',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  'openai.agents.run',
  '  openai.agents.agent:TriageAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  '    openai.agents.handoff',
  '  openai.agents.agent:Specialist',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  'openai.agents.run',
  '  openai.agents.mcp_tools',
  '    temporal:startActivity:test-mcp-list-tools',
  '      temporal:executeActivity',
  '  openai.agents.agent:McpAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  '    openai.agents.function:lookup',
  '      temporal:startActivity:test-mcp-call-tool-v2',
  '        temporal:executeActivity',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  'openai.agents.run',
  '  openai.agents.agent:GuardedAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
  '    openai.agents.guardrail:profanity-check',
  '    openai.agents.guardrail:length-check',
  'openai.agents.run',
  '  openai.agents.agent:ChildAgent',
  '    openai.agents.generation',
  '      temporal:startActivity:invokeModelActivity',
  '        temporal:executeActivity',
];

const EXPECTED_AGENT_HIERARCHY_WITHOUT_TEMPORAL_SPANS = [
  'openai.agents.run',
  '  openai.agents.agent:BasicAgent',
  '    openai.agents.generation',
  'openai.agents.run',
  '  openai.agents.agent:LocalActivityAgent',
  '    openai.agents.generation',
  '    openai.agents.function:calculateSum',
  '    openai.agents.generation',
  'openai.agents.run',
  '  openai.agents.agent:EchoToolAgent',
  '    openai.agents.generation',
  '    openai.agents.function:echo',
  '    openai.agents.generation',
  'openai.agents.run',
  '  openai.agents.agent:WeatherToolAgent',
  '    openai.agents.generation',
  '    openai.agents.function:getWeather',
  '    openai.agents.generation',
  'openai.agents.run',
  '  openai.agents.agent:MultiTurnAgent',
  '    openai.agents.generation',
  '    openai.agents.function:getWeather',
  '    openai.agents.generation',
  '    openai.agents.function:getWeather',
  '    openai.agents.generation',
  'openai.agents.run',
  '  openai.agents.agent:TriageAgent',
  '    openai.agents.generation',
  '    openai.agents.handoff',
  '  openai.agents.agent:Specialist',
  '    openai.agents.generation',
  'openai.agents.run',
  '  openai.agents.mcp_tools',
  '  openai.agents.agent:McpAgent',
  '    openai.agents.generation',
  '    openai.agents.function:lookup',
  '    openai.agents.generation',
  'openai.agents.run',
  '  openai.agents.agent:GuardedAgent',
  '    openai.agents.generation',
  '    openai.agents.guardrail:profanity-check',
  '    openai.agents.guardrail:length-check',
  'openai.agents.run',
  '  openai.agents.agent:ChildAgent',
  '    openai.agents.generation',
];

if (RUN_INTEGRATION_TESTS) {
  test.serial('comprehensive agent workflow with addTemporalSpans=true', async (t) => {
    Runtime.install({});
    try {
      const otel = createOtelContext();
      const mcpProvider = createMcpProvider();
      const agentsPlugin = new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => comprehensiveResponses()),
        mcpServerProviders: [mcpProvider],
        interceptorOptions: { addTemporalSpans: true },
      });

      const env = await createTestWorkflowEnvironment();
      try {
        const taskQueue = `tracing-with-${uuid4()}`;
        const workflowBundle = await bundleWorkflowCode({
          ...bundlerOptions,
          workflowsPath: require.resolve('./workflows/openai-agents-tracing'),
          plugins: [otel.otelPlugin, agentsPlugin],
          logger: new DefaultLogger('WARN'),
        });

        // Phase 1: start workflow, poll until signal wait, shut down worker
        const worker1 = await Worker.create({
          connection: env.nativeConnection,
          workflowBundle,
          taskQueue,
          activities: agentActivities,
          plugins: [otel.otelPlugin, agentsPlugin],
          maxCachedWorkflows: 0,
        });

        const workflowId = `comprehensive-with-${uuid4()}`;
        const handle = await env.client.workflow.start(comprehensiveAgentWorkflow, {
          taskQueue,
          workflowId,
          args: ['test-prompt'],
        });

        await worker1.runUntil(async () => {
          for (let i = 0; i < 60; i++) {
            try {
              const waiting = await handle.query<boolean>('isWaitingForSignal');
              if (waiting) return;
            } catch {
              // Query not yet available
            }
            await new Promise((r) => setTimeout(r, 1000));
          }
          throw new Error('Workflow never reached signal wait');
        });

        // Phase 2: fresh worker, send signal + updates, await result
        const agentsPlugin2 = new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider(() => comprehensiveResponses()),
          mcpServerProviders: [createMcpProvider()],
          interceptorOptions: { addTemporalSpans: true },
        });
        const worker2 = await Worker.create({
          connection: env.nativeConnection,
          workflowBundle,
          taskQueue,
          activities: agentActivities,
          plugins: [otel.otelPlugin, agentsPlugin2],
          maxCachedWorkflows: 0,
        });

        const result = await worker2.runUntil(async () => {
          await handle.signal('resume', 'signal-value');
          await handle.executeUpdate('unvalidatedUpdate', { args: ['uv'] });
          await handle.executeUpdate('validatedUpdate', { args: ['vv'] });
          return handle.result();
        });

        t.true(result.startsWith('done:'), `Expected result to start with 'done:', got: ${result}`);

        await otel.provider.shutdown();
        otelApi.trace.disable();

        const traces = buildSpanTree(otel.spans);

        // Log full hierarchy for debugging
        t.log('Traces (with temporal spans):');
        for (const trace of traces) {
          for (const line of trace) {
            t.log(line);
          }
          t.log('---');
        }

        // Exact agent hierarchy assertion — filters out OTel plugin spans
        // to avoid sensitivity to their nondeterministic ordering.
        const agentHierarchy = extractAgentHierarchy(otel.spans);
        t.log('Agent hierarchy (with temporal spans):');
        for (const line of agentHierarchy) t.log(line);
        t.deepEqual(agentHierarchy, EXPECTED_AGENT_HIERARCHY_WITH_TEMPORAL_SPANS);

        // Exact span count assertions — locks in the number of spans per
        // type. If replay causes duplicate emission, these counts would
        // double. maxCachedWorkflows: 0 forces replay on every workflow task.
        const allNames = otel.spans.map((s) => s.name);

        // 22 = sum across the expected hierarchy: Basic(1) + LocalActivity(3) +
        //   Echo(2) + Weather(3) + MultiTurn(5) + Triage+Specialist(2) +
        //   Mcp(4) + Guarded(1) + Child(1).
        const executeActivityCount = allNames.filter((n) => n === 'temporal:executeActivity').length;
        t.is(executeActivityCount, 22, `Expected 22 temporal:executeActivity, got ${executeActivityCount}`);

        // 9 openai.agents.run: one per runner.run() call (Basic, LocalActivity,
        // Echo, Weather, MultiTurn, Triage, Mcp, Guarded, Child).
        const agentRunCount = allNames.filter((n) => n === 'openai.agents.run').length;
        t.is(agentRunCount, 9, `Expected 9 openai.agents.run, got ${agentRunCount}`);

        // 16 = one generation span per LLM call: Basic(1) + LocalActivity(2) +
        //   Echo(2) + Weather(2) + MultiTurn(3) + Triage(1) + Specialist(1) +
        //   Mcp(2) + Guarded(1) + Child(1).
        const generationCount = allNames.filter((n) => n === 'openai.agents.generation').length;
        t.is(generationCount, 16, `Expected 16 openai.agents.generation, got ${generationCount}`);
      } finally {
        await env.teardown();
      }
    } finally {
      await Runtime._instance?.shutdown();
    }
  });

  test.serial('comprehensive agent workflow with addTemporalSpans=false', async (t) => {
    Runtime.install({});
    try {
      const otel = createOtelContext();
      const mcpProvider = createMcpProvider();
      // addTemporalSpans defaults to false, so omitting it tests the default
      const agentsPlugin = new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => comprehensiveResponses()),
        mcpServerProviders: [mcpProvider],
      });

      const env = await createTestWorkflowEnvironment();
      try {
        const taskQueue = `tracing-without-${uuid4()}`;
        const workflowBundle = await bundleWorkflowCode({
          ...bundlerOptions,
          workflowsPath: require.resolve('./workflows/openai-agents-tracing'),
          plugins: [otel.otelPlugin, agentsPlugin],
          logger: new DefaultLogger('WARN'),
        });

        // Phase 1: start workflow, poll until signal wait, shut down worker
        const worker1 = await Worker.create({
          connection: env.nativeConnection,
          workflowBundle,
          taskQueue,
          activities: agentActivities,
          plugins: [otel.otelPlugin, agentsPlugin],
          maxCachedWorkflows: 0,
        });

        const workflowId = `comprehensive-without-${uuid4()}`;
        const handle = await env.client.workflow.start(comprehensiveAgentWorkflowNoSpans, {
          taskQueue,
          workflowId,
          args: ['test-prompt'],
        });

        await worker1.runUntil(async () => {
          for (let i = 0; i < 60; i++) {
            try {
              const waiting = await handle.query<boolean>('isWaitingForSignal');
              if (waiting) return;
            } catch {
              // Query not yet available
            }
            await new Promise((r) => setTimeout(r, 1000));
          }
          throw new Error('Workflow never reached signal wait');
        });

        // Phase 2: fresh worker, send signal + updates, await result
        const agentsPlugin2 = new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider(() => comprehensiveResponses()),
          mcpServerProviders: [createMcpProvider()],
        });
        const worker2 = await Worker.create({
          connection: env.nativeConnection,
          workflowBundle,
          taskQueue,
          activities: agentActivities,
          plugins: [otel.otelPlugin, agentsPlugin2],
          maxCachedWorkflows: 0,
        });

        const result = await worker2.runUntil(async () => {
          await handle.signal('resume', 'signal-value');
          await handle.executeUpdate('unvalidatedUpdate', { args: ['uv'] });
          await handle.executeUpdate('validatedUpdate', { args: ['vv'] });
          return handle.result();
        });

        t.true(result.startsWith('done:'), `Expected result to start with 'done:', got: ${result}`);

        await otel.provider.shutdown();
        otelApi.trace.disable();

        const traces = buildSpanTree(otel.spans);

        // Log full hierarchy for debugging
        t.log('Traces (without temporal spans):');
        for (const trace of traces) {
          for (const line of trace) {
            t.log(line);
          }
          t.log('---');
        }

        // Exact agent hierarchy assertion
        const agentHierarchy = extractAgentHierarchy(otel.spans);
        t.log('Agent hierarchy (without temporal spans):');
        for (const line of agentHierarchy) t.log(line);
        t.deepEqual(agentHierarchy, EXPECTED_AGENT_HIERARCHY_WITHOUT_TEMPORAL_SPANS);

        // No temporal:* spans from the agent interceptors should exist
        // (OpenTelemetryPlugin's own Temporal spans like RunWorkflow/StartActivity still appear
        // since they're from the SDK's OTel interceptor, not the agent interceptor)
        const agentTemporalSpans = otel.spans.map((s) => s.name).filter((n) => n.startsWith('temporal:'));
        t.is(agentTemporalSpans.length, 0, `Should have no temporal:* agent spans, got: ${agentTemporalSpans.join(', ')}`);
      } finally {
        await env.teardown();
      }
    } finally {
      await Runtime._instance?.shutdown();
    }
  });
  test.serial('multi-workflow config isolation under reuseV8Context (truly interleaved)', async (t) => {
    // Strengthened version of the config isolation test (Item 34). The previous
    // version ran A then B back-to-back, which lets a naive `clear-on-finish`
    // implementation pass. This version is TRULY INTERLEAVED:
    //
    // 1. Start A (addTemporalSpans=true, waitForSignal=true) — it populates
    //    its store and blocks on a signal.
    // 2. Start B (addTemporalSpans=false, waitForSignal=false) — runs to
    //    completion with different config.
    // 3. Signal A to resume — A reads its config AFTER B has completed.
    //
    // A naive single-variable implementation (`let currentConfig`) would fail
    // because B's write overwrites A's value while A is still alive.
    Runtime.install({});
    try {
      const agentsPlugin = new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      });

      const env = await createTestWorkflowEnvironment();
      try {
        const taskQueue = `config-isolation-${uuid4()}`;
        const workflowBundle = await bundleWorkflowCode({
          ...bundlerOptions,
          workflowsPath: require.resolve('./workflows/openai-agents-tracing'),
          plugins: [agentsPlugin],
          logger: new DefaultLogger('WARN'),
        });

        const worker = await Worker.create({
          connection: env.nativeConnection,
          workflowBundle,
          taskQueue,
          activities: agentActivities,
          plugins: [agentsPlugin],
          maxCachedWorkflows: 0,
          reuseV8Context: true,
        });

        await worker.runUntil(async () => {
          // Step 1: Start A (blocks on signal partway through)
          const handleA = await env.client.workflow.start(configIsolationWorkflow, {
            taskQueue,
            workflowId: `config-isolation-a-${uuid4()}`,
            args: [true, true], // addTemporalSpans=true, waitForSignal=true
          });

          // Give A time to populate its store and block on the signal
          await new Promise((r) => setTimeout(r, 3000));

          // Step 2: Start B (different config), run to completion while A is alive
          const resultB = await env.client.workflow.execute(configIsolationWorkflow, {
            taskQueue,
            workflowId: `config-isolation-b-${uuid4()}`,
            args: [false, false], // addTemporalSpans=false, waitForSignal=false
          });

          // Step 3: Signal A to resume — A reads its config AFTER B completed
          await handleA.signal('proceed');
          const resultA = await handleA.result();

          // Assertions: each workflow observed its OWN config, not the other's.
          // A naive single-var impl would have B's `false` overwrite A's `true`.
          t.is(resultA, true, 'Workflow A should observe addTemporalSpans=true (not overwritten by B)');
          t.is(resultB, false, 'Workflow B should observe addTemporalSpans=false (not leaked from A)');
        });
      } finally {
        await env.teardown();
      }
    } finally {
      await Runtime._instance?.shutdown();
    }
  });

  // --- Item 26: workflow-to-workflow signalWorkflow trace propagation E2E ---

  test.serial('signalWorkflow trace propagation: OTel spans nest correctly', async (t) => {
    Runtime.install({});
    try {
      const otel = createOtelContext();
      const agentsPlugin = new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('x'), textResponse('x')]),
        interceptorOptions: { addTemporalSpans: true },
      });

      const env = await createTestWorkflowEnvironment();
      try {
        const taskQueue = `signal-trace-e2e-${uuid4()}`;
        const workflowBundle = await bundleWorkflowCode({
          ...bundlerOptions,
          workflowsPath: require.resolve('./workflows/openai-agents-tracing'),
          plugins: [otel.otelPlugin, agentsPlugin],
          logger: new DefaultLogger('WARN'),
        });

        const worker = await Worker.create({
          connection: env.nativeConnection,
          workflowBundle,
          taskQueue,
          activities: agentActivities,
          plugins: [otel.otelPlugin, agentsPlugin],
          maxCachedWorkflows: 0,
        });

        const result = await worker.runUntil(async () => {
          return env.client.workflow.execute(signalTraceParentWorkflow, {
            taskQueue,
            workflowId: `signal-trace-e2e-${uuid4()}`,
            workflowExecutionTimeout: '30 seconds',
          });
        });

        // Agent-SDK-level assertion: trace IDs should match
        t.truthy(result.parentTraceId, 'Parent should have a trace ID');
        t.not(result.signalTraceId, 'NO_SIGNAL_TRACE', 'Signal handler should have restored trace context');
        t.is(
          result.signalTraceId,
          result.parentTraceId,
          'Signal handler traceId must match parent traceId (proves signal propagation)'
        );

        await otel.provider.shutdown();
        otelApi.trace.disable();

        // OTel span hierarchy assertion: the key spans should exist
        const allNames = otel.spans.map((s) => s.name);
        t.log('All OTel span names:', allNames);

        // temporal:signalWorkflow should exist (from outbound interceptor)
        t.true(
          allNames.includes('temporal:signalWorkflow'),
          `Should have temporal:signalWorkflow span, got: ${allNames.join(', ')}`
        );

        // temporal:handleSignal should exist (from inbound interceptor)
        t.true(
          allNames.includes('temporal:handleSignal'),
          `Should have temporal:handleSignal span, got: ${allNames.join(', ')}`
        );

        // The handleSignal span should nest under signalWorkflow:
        // find the signalWorkflow span and handleSignal span, verify parent-child
        const signalSpan = otel.spans.find((s) => s.name === 'temporal:signalWorkflow');
        const handleSpan = otel.spans.find((s) => s.name === 'temporal:handleSignal');

        // Both spans are guaranteed to exist by the t.true() assertions above;
        // use non-null assertions so a missing span throws instead of silently skipping.
        const signalSpan_ = signalSpan!;
        const handleSpan_ = handleSpan!;

        // Both should be in the same OTel trace
        t.is(
          handleSpan_.spanContext().traceId,
          signalSpan_.spanContext().traceId,
          'handleSignal and signalWorkflow should share the same OTel trace ID'
        );
        t.is(
          handleSpan_.parentSpanId,
          signalSpan_.spanContext().spanId,
          'handleSignal must nest directly under signalWorkflow'
        );
        t.log('signalWorkflow spanId:', signalSpan_.spanContext().spanId);
        t.log('handleSignal parentSpanId:', handleSpan_.parentSpanId);
      } finally {
        await env.teardown();
      }
    } finally {
      await Runtime._instance?.shutdown();
    }
  });

  // --- Item 33: idempotency flag test under reuseV8Context ---

  test.serial('ensure* idempotency: processor registered once across multiple workflows under reuseV8Context', async (t) => {
    // Under reuseV8Context=true, the module-level idempotency flags
    // (processorRegistered, deterministicIdsInstalled) persist across workflows
    // sharing the same V8 isolate. If the flags weren't working, the processor
    // would be registered twice, causing duplicate OTel spans for each agent event.
    //
    // The test runs two workflows back-to-back on the same worker, collects
    // OTel spans, and asserts exact span counts match single-registration behavior.
    Runtime.install({});
    try {
      const otel = createOtelContext();
      const agentsPlugin = new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('done-a'), textResponse('done-b')]),
        interceptorOptions: { addTemporalSpans: true },
      });

      const env = await createTestWorkflowEnvironment();
      try {
        const taskQueue = `idempotency-${uuid4()}`;
        const workflowBundle = await bundleWorkflowCode({
          ...bundlerOptions,
          workflowsPath: require.resolve('./workflows/openai-agents-tracing'),
          plugins: [otel.otelPlugin, agentsPlugin],
          logger: new DefaultLogger('WARN'),
        });

        const worker = await Worker.create({
          connection: env.nativeConnection,
          workflowBundle,
          taskQueue,
          activities: agentActivities,
          plugins: [otel.otelPlugin, agentsPlugin],
          maxCachedWorkflows: 0,
          reuseV8Context: true,
        });

        await worker.runUntil(async () => {
          // Run two workflows back-to-back — they share the V8 isolate
          const resultA = await env.client.workflow.execute(idempotencyFlagWorkflow, {
            taskQueue,
            workflowId: `idempotency-a-${uuid4()}`,
            args: ['prompt-a'],
            workflowExecutionTimeout: '30 seconds',
          });
          t.is(resultA, 'done-a');

          const resultB = await env.client.workflow.execute(idempotencyFlagWorkflow, {
            taskQueue,
            workflowId: `idempotency-b-${uuid4()}`,
            args: ['prompt-b'],
            workflowExecutionTimeout: '30 seconds',
          });
          t.is(resultB, 'done-b');
        });

        await otel.provider.shutdown();
        otelApi.trace.disable();

        const allNames = otel.spans.map((s) => s.name);
        t.log('All OTel span names for idempotency test:', allNames);

        // Each workflow does exactly one runner.run() → one openai.agents.run span.
        // If the processor were registered twice, each agent event would produce
        // two OTel spans, doubling the count.
        const agentRunCount = allNames.filter((n) => n === 'openai.agents.run').length;
        t.is(agentRunCount, 2, 'Should have exactly 2 openai.agents.run spans (one per workflow, not doubled)');

        // Each workflow's agent has one generation → one openai.agents.generation span.
        const generationCount = allNames.filter((n) => n === 'openai.agents.generation').length;
        t.is(generationCount, 2, 'Should have exactly 2 openai.agents.generation spans (one per workflow, not doubled)');

        // Each generation triggers one invokeModelActivity → one temporal:startActivity span.
        const startActivityCount = allNames.filter((n) => n === 'temporal:startActivity:invokeModelActivity').length;
        t.is(startActivityCount, 2, 'Should have exactly 2 temporal:startActivity:invokeModelActivity spans (not doubled)');
      } finally {
        await env.teardown();
      }
    } finally {
      await Runtime._instance?.shutdown();
    }
  });

  // --- Item 35: agent trace context survives continueAsNew ---

  test.serial('agent trace context survives continueAsNew', async (t) => {
    Runtime.install({});
    try {
      const otel = createOtelContext();
      const agentsPlugin = new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        interceptorOptions: { addTemporalSpans: true },
      });

      const env = await createTestWorkflowEnvironment();
      try {
        const taskQueue = `trace-continue-as-new-${uuid4()}`;
        const workflowBundle = await bundleWorkflowCode({
          ...bundlerOptions,
          workflowsPath: require.resolve('./workflows/openai-agents-tracing'),
          plugins: [otel.otelPlugin, agentsPlugin],
          logger: new DefaultLogger('WARN'),
        });

        const worker = await Worker.create({
          connection: env.nativeConnection,
          workflowBundle,
          taskQueue,
          activities: agentActivities,
          plugins: [otel.otelPlugin, agentsPlugin],
          maxCachedWorkflows: 0,
        });

        const result = await worker.runUntil(async () => {
          return env.client.workflow.execute(traceContinueAsNewWorkflow, {
            taskQueue,
            workflowId: `trace-continue-as-new-${uuid4()}`,
            args: [0], // Start at iteration 0
            workflowExecutionTimeout: '30 seconds',
          });
        });

        t.truthy(result.originalTraceId, 'Should have captured the original trace ID');
        t.not(result.originalTraceId, 'NO_MEMO', 'Original trace ID should be available via memo');
        t.not(result.continuedTraceId, 'NO_TRACE', 'Continued execution should have restored trace context');
        t.is(
          result.continuedTraceId,
          result.originalTraceId,
          'Trace ID must survive continueAsNew — proves __openai_span header propagation'
        );

        await otel.provider.shutdown();
        otelApi.trace.disable();
      } finally {
        await env.teardown();
      }
    } finally {
      await Runtime._instance?.shutdown();
    }
  });
}

// --- Unit tests: createTracerProvider (Item 29) ---

test('createTracerProvider returns a ReplaySafeTracerProvider', (t) => {
  const provider = createTracerProvider();
  t.true(provider instanceof ReplaySafeTracerProvider, 'Returned provider should be instanceof ReplaySafeTracerProvider');
});

test('createTracerProvider exposes a TemporalIdGenerator', (t) => {
  const provider = createTracerProvider();
  t.true(
    provider.temporalIdGenerator instanceof TemporalIdGenerator,
    'temporalIdGenerator should be a TemporalIdGenerator instance'
  );
});

test('createTracerProvider with default options uses library defaults', (t) => {
  const provider = createTracerProvider();
  t.truthy(provider.temporalIdGenerator, 'Should have a temporalIdGenerator');
  // Verify it can generate IDs (fallback to randomHex)
  const traceId = provider.temporalIdGenerator.generateTraceId();
  t.is(traceId.length, 32, 'Default trace ID should be 32 hex chars');
  const spanId = provider.temporalIdGenerator.generateSpanId();
  t.is(spanId.length, 16, 'Default span ID should be 16 hex chars');
});

test('createTracerProvider wraps a custom idGenerator', (t) => {
  const customTraceId = 'a'.repeat(32);
  const customSpanId = 'b'.repeat(16);
  const customGenerator = {
    generateTraceId: () => customTraceId,
    generateSpanId: () => customSpanId,
  };

  const provider = createTracerProvider({ idGenerator: customGenerator });
  t.true(provider instanceof ReplaySafeTracerProvider);

  // Without seeds, it should delegate to the custom generator
  t.is(
    provider.temporalIdGenerator.generateTraceId(),
    customTraceId,
    'Should delegate to custom generator when no seed is queued'
  );
  t.is(
    provider.temporalIdGenerator.generateSpanId(),
    customSpanId,
    'Should delegate to custom generator when no seed is queued'
  );

  // With seeds, seeds take priority over the custom generator
  const seededTraceId = 'c'.repeat(32);
  provider.temporalIdGenerator.seedTraceId(seededTraceId);
  t.is(
    provider.temporalIdGenerator.generateTraceId(),
    seededTraceId,
    'Seeded trace ID should take priority over custom generator'
  );
});

// --- Unit test: loud error when global provider is not ReplaySafeTracerProvider ---

test('throws descriptive error when global TracerProvider is not ReplaySafeTracerProvider', (t) => {
  // Register a plain BasicTracerProvider (not via createTracerProvider)
  const plainProvider = new BasicTracerProvider();
  otelApi.trace.setGlobalTracerProvider(plainProvider);

  try {
    // Access the internal module via resolved absolute path to bypass
    // the package exports map. This is intentional — the constructor
    // check lives in an abstract base class not exported publicly.
    const basePath = require.resolve('@temporalio/openai-agents');
    const processorPath = basePath.replace(/lib[/\\]index\.js$/, 'lib/common/base-tracing-processor.js');
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { BaseAgentTracingProcessor } = require(processorPath);

    class TestProcessor extends BaseAgentTracingProcessor {
      getEntry() {
        return undefined;
      }
      setEntry() {}
      deleteEntry() {}
      *allEntries(): Iterable<unknown> {}
      clearAllEntries() {}
    }

    const err = t.throws(() => new TestProcessor(), {
      instanceOf: Error,
    });
    t.true(
      err!.message.includes('ReplaySafeTracerProvider'),
      'Error message should mention ReplaySafeTracerProvider'
    );
    t.true(
      err!.message.includes('createTracerProvider()'),
      'Error message should mention the factory function'
    );
  } finally {
    otelApi.trace.disable();
  }
});
