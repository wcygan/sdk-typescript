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
} from '@temporalio/openai-agents';
import { DefaultLogger, Runtime, bundleWorkflowCode } from '@temporalio/worker';
import {
  comprehensiveAgentWorkflow,
  comprehensiveAgentWorkflowNoSpans,
  tracingChildWorkflow,
  tracingChildWorkflowNoSpans,
  configIsolationWorkflow,
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
  test.serial('multi-workflow config isolation under reuseV8Context', async (t) => {
    // Guards against per-workflow plugin config leaking between workflows that
    // share a V8 isolate. Before Item 21, plugin config was stored on
    // globalThis, which silently leaked between concurrent workflows under
    // `reuseV8Context: true`. With the per-workflow plugin-config-store, each
    // workflow sees only its own config.
    //
    // The test runs A and B CONCURRENTLY: A blocks on a signal while B runs to
    // completion with a different config. After B completes, A is signaled to
    // resume and read its own config. A naive single-variable implementation
    // (`let currentConfig`) would fail because B's write overwrites A's value
    // while A is still alive.
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
          // Workflow A: addTemporalSpans = true, waits for signal before reading config
          const handleA = await env.client.workflow.start(configIsolationWorkflow, {
            taskQueue,
            workflowId: `config-isolation-a-${uuid4()}`,
            args: [true, true], // addTemporalSpans=true, waitForSignal=true
          });

          // Give A time to populate its store and block on the signal
          await new Promise((r) => setTimeout(r, 3000));

          // Workflow B: addTemporalSpans = false, runs to completion immediately
          const resultB = await env.client.workflow.execute(configIsolationWorkflow, {
            taskQueue,
            workflowId: `config-isolation-b-${uuid4()}`,
            args: [false, false], // addTemporalSpans=false, waitForSignal=false
          });

          // B completed — now signal A to resume and read its config
          await handleA.signal('proceed');
          const resultA = await handleA.result();

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
}

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
