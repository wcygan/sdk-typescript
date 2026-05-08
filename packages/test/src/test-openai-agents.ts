/**
 * Test OpenAI Agents SDK integration with Temporal workflows
 */
import { setTracingDisabled, withTrace } from '@openai/agents-core';
import { APIError } from 'openai';

// Tests opt back into agent-SDK tracing because upstream auto-disables it under NODE_ENV=test;
// the production plugin defers to upstream's default.
setTracingDisabled(false);
import {
  OpenAIAgentsPlugin,
  StatelessMCPServerProvider,
  StatefulMCPServerProvider,
  toSerializedModelResponse,
  OpenAIAgentsTraceClientInterceptor,
} from '@temporalio/openai-agents';
import {
  Client,
  WorkflowFailedError,
  WorkflowClient,
  WithStartWorkflowOperation,
  WorkflowIdConflictPolicy,
} from '@temporalio/client';
import { temporal } from '@temporalio/proto';
import {
  basicAgentWorkflow,
  toolAgentWorkflow,
  handoffAgentWorkflow,
  maxTurnsAgentWorkflow,
  multiToolAgentWorkflow,
  contextAgentWorkflow,
  rawFunctionToolWorkflow,
  runConfigStringModelWorkflow,
  localActivityAgentWorkflow,
  retryableModelWorkflow,
  agentsWorkflowErrorWorkflow,
  mcpAgentWorkflow,
  builtInToolAgentWorkflow,
  handoffInstanceWorkflow,
  cyclicHandoffWorkflow,
  promptFieldWorkflow,
  nonStringModelWorkflow,
  wrappedTemporalFailureWorkflow,
  runStreamedWorkflow,
  agentsWorkflowErrorClassCheckWorkflow,
  eventTargetListenerErrorWorkflow,
  eventTargetTargetFieldWorkflow,
  dateInResponseWorkflow,
  directToolFactoryWorkflow,
  mcpPromptsWorkflow,
  mcpFactoryArgWorkflow,
  mcpProviderWorkflow,
  summaryOverrideStringWorkflow,
  tracingUtilitiesWorkflow,
  extendedModelParamsWorkflow,
  runConfigModelOverrideCheckWorkflow,
  handoffWithRawToolWorkflow,
  handoffInstanceWithRawToolWorkflow,
  handoffMutationCheckWorkflow,
  handoffOnHandoffCallbackWorkflow,
  handoffIsEnabledFalseWorkflow,
  handoffWithCustomSchemaWorkflow,
  timeoutErrorWorkflow,
  xShouldRetryWorkflow,
  plainErrorWorkflow,
  wireRoundTripWorkflow,
  wireStrippingCheckWorkflow,
  wireVersionMismatchWorkflow,
  wireRequestSnapshotWorkflow,
  tracingSpanCaptureWorkflow,
  replaySafetyWorkflow,
  handoffCloneSnapshotWorkflow,
  concurrentTracingIsolationWorkflow,
  traceContextPropagationWorkflow,
  clientToWorkflowTraceWorkflow,
  signalTracePropagationParentWorkflow,
  childWorkflowTracePropagationParentWorkflow,
  deterministicTraceIdsWorkflow,
  statefulMcpNoWorkerWorkflow,
  statefulMcpAgentWorkflow,
  statefulMcpNotConnectedWorkflow,
  statefulMcpIsolationWorkflow,
  statefulMcpHeartbeatTimeoutWorkflow,
  statefulMcpSlowConnectHeartbeatWorkflow,
  statefulMcpReplayWorkflow,
  alsContextShapeSmokeCheckWorkflow,
  alsLeakDetectionWorkflow,
  queryTracePropagationWorkflow,
  configPropagationWorkflow,
  configUpdateWithStartWorkflow,
  configContinueAsNewWorkflow,
  configChildParentWorkflow,
  configOverridePrecedenceWorkflow,
  configFallbackWorkflow,
  configModelParamsIsolationWorkflow,
  summaryOverrideFunctionStripParentWorkflow,
} from './workflows/openai-agents';
import { helpers, makeTestFunction } from './helpers-integration';
import {
  FakeModelProvider,
  ErrorModelProvider,
  RequestCapturingModelProvider,
  ModelNameCapturingModelProvider,
  ThrowAnythingModelProvider,
  TraceCaptureModelProvider,
  textResponse,
  toolCallResponse,
  handoffResponse,
  responseWithDate,
  multiToolCallResponse,
} from './stubs/openai-agents';
import { getWeather, calculateSum } from './activities/openai-agents';
import EventType = temporal.api.enums.v1.EventType;

// --- Compile-time negative tests: plugin-side modelParams rejects function-form summaryOverride ---
// These @ts-expect-error lines document that SerializableModelActivityOptions
// (used by the plugin and client interceptor) excludes ModelSummaryProvider.
// If the type restriction is accidentally removed, TS will error on the
// "unused @ts-expect-error" directive, catching the regression at compile time.
{
  const _provider = null! as FakeModelProvider;
  // @ts-expect-error — plugin-side modelParams.summaryOverride only accepts string, not function/object
  new OpenAIAgentsPlugin({ modelProvider: _provider, modelParams: { summaryOverride: { provide: () => 'x' } } });
  // @ts-expect-error — client interceptor modelParams.summaryOverride only accepts string, not function/object
  new OpenAIAgentsTraceClientInterceptor({ modelParams: { summaryOverride: { provide: () => 'x' } } });
}

const test = makeTestFunction({
  workflowsPath: require.resolve('./workflows/openai-agents'),
  workflowInterceptorModules: [require.resolve('@temporalio/openai-agents/workflow-interceptor')],
});

test('Basic agent responds to prompt', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Hello from agent!')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result, 'Hello from agent!');
  });
});

function* toolWorkflowGenerator() {
  yield toolCallResponse('getWeather', { location: 'Tokyo' });
  yield textResponse('The weather in Tokyo is sunny, 14-20C.');
}

test('Agent can use tools backed by Temporal activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => toolWorkflowGenerator()),
      }),
    ],
    activities: {
      getWeather,
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(toolAgentWorkflow, {
      args: ['What is the weather in Tokyo?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'The weather in Tokyo is sunny, 14-20C.');

    // Verify both invokeModelActivity and getWeather appear in the workflow history
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];

    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    t.true(
      activityTypes.includes('invokeModelActivity'),
      `invokeModelActivity should be in history, got: ${activityTypes.join(', ')}`
    );
    t.true(activityTypes.includes('getWeather'), `getWeather should be in history, got: ${activityTypes.join(', ')}`);

    // Should have at least 3 activities: 2x invokeModelActivity (tool call + final response) + 1x getWeather
    t.true(
      activityScheduledEvents.length >= 3,
      `Expected at least 3 activity events, got ${activityScheduledEvents.length}`
    );
  });
});

function* handoffWorkflowGenerator() {
  // Turn 1: TriageAgent decides to hand off to WeatherSpecialist
  yield handoffResponse('transfer_to_WeatherSpecialist');
  // Turn 2: WeatherSpecialist responds with text
  yield textResponse('Sunny day!');
}

test('Agent can hand off to other agents', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => handoffWorkflowGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(handoffAgentWorkflow, {
      args: ['What is the weather in Tokyo?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.true(result.includes('Sunny'), `Expected output to contain 'Sunny', got: ${result}`);

    // Verify the handoff happened by checking activity history
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];

    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    // Should have 2 invokeModelActivity calls: one for triage agent, one for weather specialist
    const modelCalls = activityTypes.filter((name) => name === 'invokeModelActivity');
    t.true(
      modelCalls.length >= 2,
      `Expected at least 2 invokeModelActivity calls for handoff, got ${modelCalls.length}`
    );
  });
});

test('Agent respects max turns limit', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Single turn response')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(maxTurnsAgentWorkflow, {
      args: ['Hello', 1],
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result.output, 'Single turn response');
    t.true(result.turnCount <= 1, `Expected turnCount <= 1, got ${result.turnCount}`);
  });
});

test('Model invocations are scheduled as activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Activity check')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    await handle.result();

    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];

    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    t.true(
      activityTypes.includes('invokeModelActivity'),
      `invokeModelActivity should be scheduled as an activity, got: ${activityTypes.join(', ')}`
    );
  });
});

test('Handles model errors gracefully', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  // Create an APIError with a 400 status (non-retryable)
  const modelError = new APIError(400, undefined, 'Model API error', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(modelError),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });

    // Verify the error chain contains our error message
    t.truthy(err, 'Expected WorkflowFailedError');
    const fullMessage = String(err!.cause?.cause ?? err!.cause ?? err);
    t.true(
      fullMessage.includes('Model API error'),
      `Expected error chain to contain 'Model API error', got: ${fullMessage}`
    );

    // Verify error chain preserves classification
    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.is(
      failure?.applicationFailureInfo?.type,
      'ModelInvocationError.BadRequest',
      `Expected error type 'ModelInvocationError.BadRequest' for 400, got: ${failure?.applicationFailureInfo?.type}`
    );
    t.true(
      failure?.applicationFailureInfo?.nonRetryable === true,
      `Expected nonRetryable=true for 400, got nonRetryable=${failure?.applicationFailureInfo?.nonRetryable}`
    );
    t.true(
      failure?.message?.includes('Model API error') === true,
      `Expected original message preserved in failure, got: ${failure?.message}`
    );
  });
});

function* multiToolGenerator() {
  yield toolCallResponse('getWeather', { location: 'Tokyo' });
  yield toolCallResponse('calculateSum', { a: 5, b: 3 });
  yield textResponse('Weather in Tokyo is sunny and 5+3=8.');
}

test('Agent with multiple tools', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => multiToolGenerator()),
      }),
    ],
    activities: {
      getWeather,
      calculateSum,
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(multiToolAgentWorkflow, {
      args: ['What is the weather in Tokyo and what is 5+3?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Weather in Tokyo is sunny and 5+3=8.');

    // Verify both tool activities appear in history
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];

    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    t.true(activityTypes.includes('getWeather'), `getWeather should be in history, got: ${activityTypes.join(', ')}`);
    t.true(
      activityTypes.includes('calculateSum'),
      `calculateSum should be in history, got: ${activityTypes.join(', ')}`
    );
  });
});

test('Agent workflow with typed context', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Hello user-123!')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(contextAgentWorkflow, {
      args: ['Hello', 'user-123'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result, 'Hello user-123!');
  });
});

test('Raw function tool is rejected with clear error', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach here')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(rawFunctionToolWorkflow, {
      args: ['What is the weather?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });

    t.truthy(err, 'Expected WorkflowFailedError');
    const fullMessage = String(err!.cause?.cause ?? err!.cause ?? err);
    t.true(fullMessage.includes('activityAsTool'), `Expected error to mention activityAsTool, got: ${fullMessage}`);
  });
});

test('RunConfig.model string override works', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Model override response')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(runConfigStringModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result, 'Model override response');
  });
});

test('Local activity mode uses local activities for model calls', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Local activity response')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(localActivityAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Local activity response');

    const { events } = await handle.fetchHistory();

    // Local activities appear as MarkerRecorded events (marker name "core_local_activity"),
    // not as ActivityTaskScheduled events
    const markerEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_MARKER_RECORDED) ?? [];
    t.true(
      markerEvents.length > 0,
      `Expected MarkerRecorded events for local activities in history, got ${markerEvents.length}`
    );

    // Should NOT have regular activity scheduled events for model invocation
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    const modelActivities = activityScheduledEvents.filter(
      (e) => e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.is(modelActivities.length, 0, `Expected no regular invokeModelActivity, got ${modelActivities.length}`);
  });
});

test('Retryable 429 error is classified as retryable (nonRetryable=false)', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error429 = new APIError(429, undefined, 'Rate limit exceeded', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error429),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(retryableModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    // Workflow should fail after all retry attempts are exhausted
    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });

    t.truthy(err, 'Expected WorkflowFailedError');
    const fullMessage = String(err!.cause?.cause ?? err!.cause ?? err);
    t.true(
      fullMessage.includes('Rate limit exceeded'),
      `Expected error chain to contain 'Rate limit exceeded', got: ${fullMessage}`
    );

    // Verify the failure is classified as retryable
    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    const nonRetryable = failure?.applicationFailureInfo?.nonRetryable;
    t.falsy(
      nonRetryable,
      `Expected 429 to be classified as retryable (nonRetryable=false), got nonRetryable=${nonRetryable}`
    );
    const failureType = failure?.applicationFailureInfo?.type;
    t.is(failureType, 'ModelInvocationError.RateLimit');
  });
});

test('Non-retryable 400 error fails without retry', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error400 = new APIError(400, undefined, 'Bad request: invalid prompt', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error400),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });

    t.truthy(err, 'Expected WorkflowFailedError');
    const fullMessage = String(err!.cause?.cause ?? err!.cause ?? err);
    t.true(
      fullMessage.includes('Bad request: invalid prompt'),
      `Expected error chain to contain 'Bad request: invalid prompt', got: ${fullMessage}`
    );

    // Verify only 1 activity attempt — non-retryable errors should not be retried
    const { events } = await handle.fetchHistory();
    const activityStartedEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED) ?? [];
    t.is(
      activityStartedEvents.length,
      1,
      `Expected exactly 1 activity attempt (no retry), got ${activityStartedEvents.length}`
    );

    // Verify the failure is classified as non-retryable
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.true(failure?.applicationFailureInfo?.nonRetryable, 'Expected 400 to be classified as non-retryable');
  });
});

test('AgentsWorkflowError wraps non-Temporal errors', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach here')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentsWorkflowErrorWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });

    t.truthy(err, 'Expected WorkflowFailedError');
    // Error chain: err.cause = ApplicationFailure(type='AgentsWorkflowError'),
    // err.cause.cause = original Error. No intermediate AgentsWorkflowError wrapper.
    const wrappedMessage = String(err!.cause);
    t.true(
      wrappedMessage.includes('Agent workflow failed'),
      `Expected wrapper message to contain 'Agent workflow failed', got: ${wrappedMessage}`
    );
    t.true(
      wrappedMessage.includes('Instructions evaluation failed'),
      `Expected wrapper to contain original error message, got: ${wrappedMessage}`
    );
  });
});

// --- Stateless MCP ---

function* mcpToolWorkflowGenerator() {
  // Turn 1: model calls the MCP tool "get_time"
  yield toolCallResponse('get_time', {});
  // Turn 2: model returns a text response incorporating the tool result
  yield textResponse('The current time is 2026-01-01T00:00:00Z.');
}

test('Stateless MCP server delegates listTools and callTool to activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => mcpToolWorkflowGenerator()),
      }),
    ],
    activities: {
      'testMcp-list-tools': async () => {
        return [
          {
            name: 'get_time',
            description: 'Returns current time',
            inputSchema: {
              type: 'object' as const,
              properties: {},
              required: [] as string[],
              additionalProperties: false,
            },
          },
        ];
      },
      'testMcp-call-tool-v2': async (_input: { toolName: string; args: Record<string, unknown> | null }) => {
        return [{ type: 'text', text: '2026-01-01T00:00:00Z' }];
      },
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpAgentWorkflow, {
      args: ['What time is it?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'The current time is 2026-01-01T00:00:00Z.');

    // Verify MCP activities appear in the workflow history
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];

    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    t.true(
      activityTypes.includes('testMcp-list-tools'),
      `testMcp-list-tools should be in history, got: ${activityTypes.join(', ')}`
    );
    t.true(
      activityTypes.includes('testMcp-call-tool-v2'),
      `testMcp-call-tool-v2 should be in history, got: ${activityTypes.join(', ')}`
    );
    t.true(
      activityTypes.includes('invokeModelActivity'),
      `invokeModelActivity should be in history, got: ${activityTypes.join(', ')}`
    );
  });
});

// --- Built-in tools pass-through ---

test('Built-in tools pass through without serialization error', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse("I could search but won't")]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(builtInToolAgentWorkflow, {
      args: ['Search for something'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, "I could search but won't");

    // Verify the model activity fired (the built-in tool survived serialization)
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];

    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    t.true(
      activityTypes.includes('invokeModelActivity'),
      `invokeModelActivity should be in history, got: ${activityTypes.join(', ')}`
    );
  });
});

// --- Bug exercise tests: handoff, cycle, prompt, model validation ---

function* handoffInstanceGenerator() {
  yield handoffResponse('transfer_to_WeatherSpecialist');
  yield textResponse('Specialist says: sunny!');
}

test('Handoff-instance handoff reaches target agent via model activity', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => handoffInstanceGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(handoffInstanceWorkflow, {
      args: ['What is the weather in Tokyo?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.true(result.includes('sunny'), `Expected output to contain 'sunny', got: ${result}`);

    // Verify at least 2 model activity calls (triage + specialist after handoff)
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    const modelCalls = activityScheduledEvents.filter(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.true(
      modelCalls.length >= 2,
      `Expected >= 2 invokeModelActivity calls (triage + specialist), got ${modelCalls.length}`
    );
  });
});

test('Cyclic handoff graph terminates without stack overflow', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('ok')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(cyclicHandoffWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '5 seconds',
    });
    t.is(result, 'ok');
  });
});

test('prompt field is forwarded through ActivityBackedModel to the activity', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const provider = new RequestCapturingModelProvider();
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: provider,
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(promptFieldWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'captured');
  });

  // After workflow completes, verify the model received the prompt field
  t.truthy(provider.lastRequest, 'Expected model to have received a request');
  const receivedPrompt = (provider.lastRequest as any)?.prompt;
  t.truthy(receivedPrompt, 'Expected prompt field to be present in model request');
  t.is(receivedPrompt?.promptId, 'pt_test', `Expected promptId 'pt_test', got: ${receivedPrompt?.promptId}`);
});

test('Non-string agent.model throws AgentsWorkflowError', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(nonStringModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });

    t.truthy(err, 'Expected WorkflowFailedError');
    const cause = err!.cause as any;
    const failureType =
      cause?.failure?.applicationFailureInfo?.type ?? cause?.applicationFailureInfo?.type ?? cause?.type;
    t.is(failureType, 'AgentsWorkflowError', `Expected type 'AgentsWorkflowError', got: ${failureType}`);

    const fullMessage = String(cause);
    t.true(fullMessage.includes('string'), `Expected error message to mention 'string', got: ${fullMessage}`);
  });
});

test('SDK-shape 429 (error.status) classified as retryable', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const sdkError429 = new APIError(429, undefined, 'Rate limit exceeded', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(sdkError429),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(retryableModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.falsy(
      failure?.applicationFailureInfo?.nonRetryable,
      'Expected SDK-shape 429 to be classified as retryable (nonRetryable=false)'
    );
  });
});

test('SDK-shape 400 (error.status) classified as non-retryable', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const sdkError400 = new APIError(400, undefined, 'Bad request: invalid parameters', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(sdkError400),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(retryableModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.true(failure?.applicationFailureInfo?.nonRetryable, 'Expected SDK-shape 400 to be classified as non-retryable');

    // Non-retryable means only 1 attempt
    const startedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED) ?? [];
    t.is(startedEvents.length, 1, `Expected 1 attempt for non-retryable, got ${startedEvents.length}`);
  });
});

test('retry-after-ms header sets nextRetryDelay on activity failure', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const headers = new Headers({ 'retry-after-ms': '5000' });
  const error429 = new APIError(429, undefined, 'Rate limited', headers);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error429),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(retryableModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');

    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    const nextRetryDelay = failure?.applicationFailureInfo?.nextRetryDelay;
    t.truthy(nextRetryDelay, 'Expected nextRetryDelay to be set from retry-after-ms header');
    const delaySec = Number(nextRetryDelay?.seconds ?? 0);
    t.is(delaySec, 5, `Expected nextRetryDelay of 5 seconds (from retry-after-ms: 5000), got: ${delaySec}s`);
  });
});

test('TemporalFailure in Error.cause is unwrapped and re-thrown', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(wrappedTemporalFailureWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });
    t.truthy(err);

    // Inner ApplicationFailure (type 'InnerFailureType') must be re-thrown directly,
    // not wrapped as 'AgentsWorkflowError'.
    const cause = err!.cause as any;
    const failureType =
      cause?.failure?.applicationFailureInfo?.type ?? cause?.applicationFailureInfo?.type ?? cause?.type;
    t.not(
      failureType,
      'AgentsWorkflowError',
      `Expected inner TemporalFailure to propagate, not be wrapped as AgentsWorkflowError`
    );
    t.is(failureType, 'InnerFailureType', `Expected failure type 'InnerFailureType', got: ${failureType}`);
  });
});

test('AgentsWorkflowError type is preserved in serialized failure', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentsWorkflowErrorWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });
    t.truthy(err);

    const cause = err!.cause as any;
    const failureType =
      cause?.failure?.applicationFailureInfo?.type ?? cause?.applicationFailureInfo?.type ?? cause?.type;
    t.is(failureType, 'AgentsWorkflowError', `Expected failure type 'AgentsWorkflowError', got: ${failureType}`);
  });
});

// Verify runner wraps errors as ApplicationFailure with original error as cause.
test('Runner wraps error as ApplicationFailure with original Error cause', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentsWorkflowErrorClassCheckWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    const info = JSON.parse(result);
    t.is(
      info.causeName,
      'Error',
      `Expected runner to throw with original Error as cause, got causeName=${info.causeName}`
    );
  });
});

test('runStreamed is not available on TemporalOpenAIRunner', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(runStreamedWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });
    t.truthy(err, 'Expected WorkflowFailedError when calling runStreamed()');
  });
});

// --- Determinism + error hygiene ---

test('Non-Error thrown value is wrapped and preserved as cause', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ThrowAnythingModelProvider('custom string error'),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });
    t.truthy(err);

    const fullMessage = String(err!.cause?.cause ?? err!.cause ?? err);
    t.true(
      fullMessage.includes('custom string error'),
      `Expected error chain to contain 'custom string error', got: ${fullMessage}`
    );

    // The key assertion: the activity failure's cause should be preserved (not undefined)
    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.truthy(failure?.cause, 'Expected non-Error value to be wrapped in Error and preserved as cause');
  });
});

test('EventTarget polyfill isolates listener errors', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(eventTargetListenerErrorWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.true(result.dispatchSucceeded, 'dispatchEvent should succeed even if a listener throws');
    t.true(result.secondListenerCalled, 'Second listener should be called even if first throws');
  });
});

test('EventTarget polyfill sets event.target and event.currentTarget', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(eventTargetTargetFieldWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.true(result.targetDefined, 'event.target should be defined (set to the EventTarget instance)');
    t.true(result.currentTargetDefined, 'event.currentTarget should be defined (set to the EventTarget instance)');
  });
});

test('429 error produces ModelInvocationError.RateLimit type', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error429 = new APIError(429, undefined, 'Rate limit exceeded', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error429),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(retryableModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    const failureType = failure?.applicationFailureInfo?.type;
    t.is(failureType, 'ModelInvocationError.RateLimit', `Expected RateLimit type for 429, got: ${failureType}`);
  });
});

test('401 error produces ModelInvocationError.Authentication type', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error401 = new APIError(401, undefined, 'Unauthorized', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error401),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    const failureType = failure?.applicationFailureInfo?.type;
    t.is(
      failureType,
      'ModelInvocationError.Authentication',
      `Expected Authentication type for 401, got: ${failureType}`
    );
  });
});

test('400 error produces ModelInvocationError.BadRequest type', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error400 = new APIError(400, undefined, 'Bad request', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error400),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    const failureType = failure?.applicationFailureInfo?.type;
    t.is(failureType, 'ModelInvocationError.BadRequest', `Expected BadRequest type for 400, got: ${failureType}`);
  });
});

test('500 error produces ModelInvocationError.ServerError type', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error500 = new APIError(500, undefined, 'Internal server error', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error500),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(retryableModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    const failureType = failure?.applicationFailureInfo?.type;
    t.is(failureType, 'ModelInvocationError.ServerError', `Expected ServerError type for 500, got: ${failureType}`);
  });
});

test('Non-Error non-object throw produces retryable failure (defers to Temporal retry policy)', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ThrowAnythingModelProvider(42),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(retryableModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const fullMessage = String(err!.cause?.cause ?? err!.cause ?? err);
    t.true(fullMessage.includes('42'), `Expected error chain to contain '42', got: ${fullMessage}`);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.falsy(
      failure?.applicationFailureInfo?.nonRetryable,
      'Expected non-APIError throw to be retryable (nonRetryable=false, defers to retry policy)'
    );
  });
});

// Temporal's default payload converter serializes via JSON.stringify.
// Date objects become ISO strings, class instances become plain objects.
// @openai/agents-core's ModelResponse uses plain JSON-safe types by default,
// so this is typically not a concern. Custom ModelProviders that emit Dates
// or class instances should pre-serialize them.
test('Date in ModelResponse is coerced to string by Temporal serialization', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([responseWithDate('Date test')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(dateInResponseWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(typeof result.hasDateField, 'boolean', 'Workflow should return hasDateField status');
    if (result.hasDateField) {
      t.is(result.dateFieldType, 'string', 'Date is coerced to ISO string by Temporal JSON serialization');
    } else {
      t.is(result.dateFieldType, 'undefined', 'Stripped custom field should have undefined type');
    }
  });
});

// --- Tool validation ---

// tool() from agents-core runs inline in the workflow — no activity boundary
function* inlineToolGenerator() {
  yield toolCallResponse('inlineTool', { input: 'hello' });
  yield textResponse('Tool said: processed: hello');
}

test('FunctionTool from tool() factory runs inline in workflow', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => inlineToolGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(directToolFactoryWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result, 'Tool said: processed: hello');
  });
});

// --- MCP prompts + provider ---

test('MCP listPrompts and getPrompt delegate to activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
    activities: {
      'testMcp-list-tools': async () => [],
      'testMcp-call-tool-v2': async () => [],
      'testMcp-list-prompts': async () => {
        return [
          { name: 'greeting', description: 'A greeting prompt' },
          { name: 'farewell', description: 'A farewell prompt' },
        ];
      },
      'testMcp-get-prompt-v2': async (input: {
        promptName: string;
        promptArguments: Record<string, unknown> | null;
      }) => {
        return {
          messages: [{ role: 'user', content: `Hello, ${(input.promptArguments as any)?.name ?? 'stranger'}!` }],
        };
      },
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpPromptsWorkflow, {
      args: ['test'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();

    // Verify listPrompts returned data
    t.is((result.prompts as any[]).length, 2, 'Expected 2 prompts from listPrompts');
    t.is((result.prompts as any[])[0].name, 'greeting');

    // Verify getPrompt returned data
    t.truthy(result.promptResult, 'Expected getPrompt to return data');
    t.is((result.promptResult as any).messages[0].content, 'Hello, World!');

    // Verify activities appeared in history
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    t.true(
      activityTypes.includes('testMcp-list-prompts'),
      `testMcp-list-prompts should be in history, got: ${activityTypes.join(', ')}`
    );
    t.true(
      activityTypes.includes('testMcp-get-prompt-v2'),
      `testMcp-get-prompt-v2 should be in history, got: ${activityTypes.join(', ')}`
    );
  });
});

function* mcpFactoryArgGenerator() {
  yield toolCallResponse('get_time', {});
  yield textResponse('The time for tenant-42 is 2026-01-01.');
}

test('factoryArgument is passed through to MCP activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  let receivedFactoryArg: unknown;
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => mcpFactoryArgGenerator()),
      }),
    ],
    activities: {
      'testMcp-list-tools': async (input: any) => {
        receivedFactoryArg = input?.factoryArgument;
        return [
          {
            name: 'get_time',
            description: 'Returns current time',
            inputSchema: {
              type: 'object' as const,
              properties: {},
              required: [] as string[],
              additionalProperties: false,
            },
          },
        ];
      },
      'testMcp-call-tool-v2': async (input: any) => {
        t.deepEqual(input.factoryArgument, { tenantId: 'tenant-42' }, 'factoryArgument should be passed to callTool');
        return [{ type: 'text', text: '2026-01-01' }];
      },
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpFactoryArgWorkflow, {
      args: ['What time is it?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.truthy(result, 'Workflow should complete successfully');

    // Verify factoryArgument was passed to listTools activity
    t.deepEqual(receivedFactoryArg, { tenantId: 'tenant-42' }, 'factoryArgument should be passed to listTools');
  });
});

function* mcpProviderGenerator() {
  yield toolCallResponse('get_data', {});
  yield textResponse('Data retrieved via provider.');
}

test('StatelessMCPServerProvider registers activities via plugin', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const mcpProvider = new StatelessMCPServerProvider('providerMcp', {
    listTools: async () => [
      {
        name: 'get_data',
        description: 'Get some data',
        inputSchema: { type: 'object' as const, properties: {}, required: [] as string[], additionalProperties: false },
      },
    ],
    callTool: async () => [{ type: 'text', text: 'provider-data-result' }],
    listPrompts: async () => [],
    getPrompt: async () => ({ messages: [] }),
  });

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => mcpProviderGenerator()),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpProviderWorkflow, {
      args: ['Get the data'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Data retrieved via provider.');

    // Verify provider-registered activities appear in history
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    t.true(
      activityTypes.includes('providerMcp-list-tools'),
      `providerMcp-list-tools should be in history, got: ${activityTypes.join(', ')}`
    );
    t.true(
      activityTypes.includes('providerMcp-call-tool-v2'),
      `providerMcp-call-tool-v2 should be in history, got: ${activityTypes.join(', ')}`
    );
  });
});

// --- Summary override ---

test('summaryOverride string is passed through to model activity', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Summary test response')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(summaryOverrideStringWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Summary test response');

    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    t.true(
      activityScheduledEvents.length >= 1,
      `Expected at least 1 activity scheduled event, got ${activityScheduledEvents.length}`
    );

    const modelEvent = activityScheduledEvents.find(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.truthy(modelEvent, 'Expected invokeModelActivity in history');
    const userMetadata = (modelEvent as any)?.userMetadata;
    t.truthy(userMetadata, 'Expected userMetadata on activity scheduled event');
    if (userMetadata) {
      const summaryPayload = userMetadata?.summary;
      t.truthy(summaryPayload, 'Expected summary payload in userMetadata');
      if (summaryPayload) {
        const summaryText = Buffer.from(summaryPayload.data).toString('utf-8');
        t.true(
          summaryText.includes('Custom model summary'),
          `Expected summary metadata to contain 'Custom model summary', got: ${summaryText}`
        );
      }
    }
  });
});

// --- Tracing utilities ---

test('Tracing utilities return correct values in workflow context', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(tracingUtilitiesWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.true(result.isInWf, 'isInWorkflow() should return true inside workflow');
    t.is(typeof result.isReplay, 'boolean', 'isReplaying() should return a boolean');
  });
});

test('Tracing processor smoke check passes with current upstream shape', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    // First workflow: triggers ensureTracingProcessorRegistered and the ALS smoke check.
    const result = await executeWorkflow(alsContextShapeSmokeCheckWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'ok', 'Smoke check should not throw — ALS context shape is compatible');

    // Second workflow on the same worker (same V8 isolate): getCurrentTrace() at
    // workflow start must not see the smoke-check sentinel. If it does, the sentinel
    // leaked via the broken upstream ALS shim.
    const leakedTraceId = await executeWorkflow(alsLeakDetectionWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(leakedTraceId, null, 'Smoke-check sentinel must not leak into subsequent workflows');
  });
});

// --- Additional model activity parameters ---

test('Extended model params (priority) pass through without error', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Extended params OK')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(extendedModelParamsWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result, 'Extended params OK');
  });
});

// --- Public testing namespace ---

test('Testing namespace exports are importable', async (t) => {
  // Verify the testing namespace is accessible from the main package
  const testing = await import('@temporalio/openai-agents/lib/testing');

  t.truthy(testing.FakeModel, 'FakeModel should be exported');
  t.truthy(testing.FakeModelProvider, 'FakeModelProvider should be exported');
  t.truthy(testing.textResponse, 'textResponse should be exported');
  t.truthy(testing.toolCallResponse, 'toolCallResponse should be exported');
  t.truthy(testing.handoffResponse, 'handoffResponse should be exported');
  t.truthy(testing.multiToolCallResponse, 'multiToolCallResponse should be exported');
  t.truthy(testing.ResponseBuilders, 'ResponseBuilders namespace should be exported');

  // Verify they work
  const response = testing.textResponse('test');
  t.truthy(response.output, 'textResponse should produce a valid ModelResponse');
});

// --- Retry and replay ---

// Verify retry policy is applied — retryState proves the server used the policy
test('Retryable 429 error exhausts retry policy (retryState=MAXIMUM_ATTEMPTS_REACHED)', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error429 = new APIError(429, undefined, 'Rate limit exceeded', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error429),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(retryableModelWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });
    t.truthy(err, 'Workflow should fail after retry policy is exhausted');

    const { events } = await handle.fetchHistory();

    // Temporal dev server reports MAX_ATTEMPTS_REACHED regardless of actual retry count;
    // asserting retryState proves the retry-policy path was taken (vs NON_RETRYABLE_FAILURE).
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one ACTIVITY_TASK_FAILED event');
    const lastFailed = failedEvents[failedEvents.length - 1];
    t.is(
      lastFailed?.activityTaskFailedEventAttributes?.retryState,
      4, // RETRY_STATE_MAXIMUM_ATTEMPTS_REACHED
      'Retry state should be MAXIMUM_ATTEMPTS_REACHED (retry policy applied, not non-retryable)'
    );
  });
});

// Parallel tool calls — single model response containing multiple function_calls
function* parallelToolCallGenerator() {
  yield multiToolCallResponse([
    { name: 'getWeather', args: { location: 'Tokyo' } },
    { name: 'calculateSum', args: { a: 5, b: 3 } },
  ]);
  yield textResponse('Weather is sunny and 5+3=8.');
}

test('Parallel tool calls in one model response', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => parallelToolCallGenerator()),
      }),
    ],
    activities: {
      getWeather,
      calculateSum,
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(multiToolAgentWorkflow, {
      args: ['What is the weather in Tokyo and what is 5+3?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Weather is sunny and 5+3=8.');

    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );

    t.true(activityTypes.includes('getWeather'), `getWeather should be scheduled, got: ${activityTypes.join(', ')}`);
    t.true(
      activityTypes.includes('calculateSum'),
      `calculateSum should be scheduled, got: ${activityTypes.join(', ')}`
    );

    // Both tool calls from a single model response + the final text response = 2 model activity calls
    const modelCalls = activityTypes.filter((name) => name === 'invokeModelActivity');
    t.is(modelCalls.length, 2, `Expected 2 invokeModelActivity calls, got ${modelCalls.length}`);

    // Verify parallel scheduling: both tool activities should be scheduled
    // in the same workflow task (same workflowTaskCompletedEventId)
    const toolEvents = activityScheduledEvents.filter((e) => {
      const name = e?.activityTaskScheduledEventAttributes?.activityType?.name;
      return name === 'getWeather' || name === 'calculateSum';
    });
    if (toolEvents.length === 2) {
      const taskId1 = (toolEvents[0]?.activityTaskScheduledEventAttributes as any)?.workflowTaskCompletedEventId;
      const taskId2 = (toolEvents[1]?.activityTaskScheduledEventAttributes as any)?.workflowTaskCompletedEventId;
      t.truthy(taskId1, 'Expected workflowTaskCompletedEventId on first tool event');
      t.deepEqual(taskId1, taskId2, 'Both tool activities should be scheduled in the same workflow task (parallel)');
    }
  });
});

// Replay smoke test — verify determinism by replaying recorded history
test('Workflow replay succeeds without determinism errors', async (t) => {
  const { createWorker, startWorkflow, runReplayHistory } = helpers(t);

  let history: temporal.api.history.v1.IHistory | undefined;

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => toolWorkflowGenerator()),
      }),
    ],
    activities: {
      getWeather,
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(toolAgentWorkflow, {
      args: ['What is the weather in Tokyo?'],
      workflowExecutionTimeout: '30 seconds',
    });

    await handle.result();
    history = (await handle.fetchHistory()) ?? undefined;
  });

  t.truthy(history, 'Should have captured workflow history');
  await runReplayHistory({}, history!);
  t.pass('Replay completed without determinism errors');
});

// Schema-invalid tool input — activityAsTool does not validate args against schema
function* schemaInvalidToolInputGenerator() {
  yield toolCallResponse('calculateSum', { x: 5, y: 3 });
  yield textResponse('The calculation returned a result.');
}

// --- runConfig.model override reaches activity ---

test('runConfig.model string override uses override model name in activity', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const provider = new ModelNameCapturingModelProvider();
  const worker = await createWorker({
    plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(runConfigModelOverrideCheckWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'captured');
  });

  t.true(
    provider.capturedModelNames.includes('override-model'),
    `Expected 'override-model' in activity, got: ${provider.capturedModelNames.join(', ')}`
  );
  t.false(
    provider.capturedModelNames.includes('original-model'),
    `Agent's original model 'original-model' should NOT be used when runConfig.model overrides it, got: ${provider.capturedModelNames.join(
      ', '
    )}`
  );
});

// --- convertAgent catches raw function tools on handoff agents ---

test('convertAgent catches raw function tool on handoff agent (Agent handoff)', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(handoffWithRawToolWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });

    t.truthy(err, 'Expected WorkflowFailedError for raw tool on handoff agent');
    const fullMessage = String(err!.cause?.cause ?? err!.cause ?? err);
    t.true(
      fullMessage.includes('raw function') || fullMessage.includes('not a tool'),
      `Expected error about raw function tool on handoff agent, got: ${fullMessage}`
    );
  });
});

test('convertAgent catches raw function tool on handoff() instance agent', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(handoffInstanceWithRawToolWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), {
      instanceOf: WorkflowFailedError,
    });

    t.truthy(err, 'Expected WorkflowFailedError for raw tool on handoff() agent');
    const fullMessage = String(err!.cause?.cause ?? err!.cause ?? err);
    t.true(
      fullMessage.includes('raw function') || fullMessage.includes('not a tool'),
      `Expected error about raw function tool, got: ${fullMessage}`
    );
  });
});

// --- Handoff mutation ---

function* handoffMutationGenerator() {
  yield handoffResponse('transfer_to_Specialist');
  yield textResponse('Specialist says hello');
}

test('convertAgent does not mutate original Handoff objects', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => handoffMutationGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(handoffMutationCheckWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const info = JSON.parse(result);
    t.false(
      info.mutated,
      `Original handoff should not be mutated. Model type was '${info.originalModelType}' before, '${info.afterModelType}' after`
    );
  });
});

// --- Error classification edge cases ---

test('408 Timeout error produces ModelInvocationError.Timeout type', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error408 = new APIError(408, undefined, 'Request timeout', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error408),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(timeoutErrorWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.is(
      failure?.applicationFailureInfo?.type,
      'ModelInvocationError.Timeout',
      `Expected Timeout type for 408, got: ${failure?.applicationFailureInfo?.type}`
    );
    t.falsy(failure?.applicationFailureInfo?.nonRetryable, 'Expected 408 to be classified as retryable');
  });
});

test('409 Conflict error produces ModelInvocationError.Conflict type', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error409 = new APIError(409, undefined, 'Conflict', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error409),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(timeoutErrorWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.is(
      failure?.applicationFailureInfo?.type,
      'ModelInvocationError.Conflict',
      `Expected Conflict type for 409, got: ${failure?.applicationFailureInfo?.type}`
    );
    t.falsy(failure?.applicationFailureInfo?.nonRetryable, 'Expected 409 to be classified as retryable');
  });
});

test('422 error produces ModelInvocationError.BadRequest type', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const error422 = new APIError(422, undefined, 'Unprocessable entity', new Headers());

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error422),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.is(
      failure?.applicationFailureInfo?.type,
      'ModelInvocationError.BadRequest',
      `Expected BadRequest type for 422, got: ${failure?.applicationFailureInfo?.type}`
    );
    t.true(failure?.applicationFailureInfo?.nonRetryable, 'Expected 422 to be classified as non-retryable');
  });
});

test('x-should-retry true overrides non-retryable 400 to retryable', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const headers = new Headers({ 'x-should-retry': 'true' });
  const error400WithRetry = new APIError(400, undefined, 'Bad request but should retry', headers);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error400WithRetry),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(xShouldRetryWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.falsy(
      failure?.applicationFailureInfo?.nonRetryable,
      'Expected x-should-retry:true to make 400 retryable (nonRetryable=false)'
    );
  });
});

test('Plain Error without HTTP status is retryable (defers to Temporal retry policy)', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(new Error('non-HTTP bug')),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(plainErrorWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.falsy(
      failure?.applicationFailureInfo?.nonRetryable,
      'Expected plain Error (no HTTP status / non-APIError) to be retryable (nonRetryable=false)'
    );
  });
});

test('x-should-retry false overrides retryable 429 to non-retryable', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const headers = new Headers({ 'x-should-retry': 'false' });
  const error429NoRetry = new APIError(429, undefined, 'Rate limit but do not retry', headers);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new ErrorModelProvider(error429NoRetry),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(basicAgentWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.truthy(err);

    const { events } = await handle.fetchHistory();
    const failedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED) ?? [];
    t.true(failedEvents.length >= 1, 'Expected at least one activity failure');
    const failure = failedEvents[0]?.activityTaskFailedEventAttributes?.failure;
    t.true(failure?.applicationFailureInfo?.nonRetryable, 'Expected x-should-retry:false to make 429 non-retryable');

    const startedEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED) ?? [];
    t.is(startedEvents.length, 1, `Expected 1 attempt for non-retryable, got ${startedEvents.length}`);
  });
});

// --- Handoff option preservation ---

function* handoffCallbackGenerator() {
  yield handoffResponse('transfer_to_CallbackSpecialist', { reason: 'weather question' });
  yield textResponse('Specialist handled it!');
}

test('Handoff onHandoff callback is preserved through convertAgent', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => handoffCallbackGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(handoffOnHandoffCallbackWorkflow, {
      args: ['What is the weather?'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.true(
      result.onHandoffCalled,
      'onHandoff callback should fire when handoff is invoked (convertAgent must preserve it)'
    );
    t.true(result.output.includes('Specialist'), `Expected output from specialist, got: ${result.output}`);
  });
});

test('Handoff isEnabled=false is preserved through convertAgent', async (t) => {
  const provider = new RequestCapturingModelProvider();
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(handoffIsEnabledFalseWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'captured');
  });

  const handoffs = (provider.lastRequest as any)?.handoffs ?? [];
  t.is(
    handoffs.length,
    0,
    `Expected 0 handoffs (isEnabled=false should hide it), got ${handoffs.length}: ${handoffs
      .map((h: any) => h.toolName)
      .join(', ')}`
  );
});

test('Handoff inputJsonSchema is preserved through convertAgent', async (t) => {
  const provider = new RequestCapturingModelProvider();
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(handoffWithCustomSchemaWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'captured');
  });

  const handoffs = (provider.lastRequest as any)?.handoffs ?? [];
  t.true(handoffs.length >= 1, 'Expected at least 1 handoff');
  const schema = handoffs[0]?.inputJsonSchema;
  t.truthy(
    schema?.properties?.reason,
    `Expected inputJsonSchema to have 'reason' property from custom schema, got: ${JSON.stringify(schema)}`
  );
});

test('Schema-invalid tool input is passed through without validation', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => schemaInvalidToolInputGenerator()),
      }),
    ],
    activities: {
      getWeather,
      calculateSum,
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(multiToolAgentWorkflow, {
      args: ['Calculate something'],
      workflowExecutionTimeout: '30 seconds',
    });

    // activityAsTool does not validate tool arguments against the JSON schema.
    // With { x: 5, y: 3 } instead of { a: number, b: number }, the calculateSum
    // activity receives undefined for a and b, producing NaN (serialized as null).
    // agents-core feeds the result back to the model, which produces a text response.
    const result = await handle.result();
    t.is(result, 'The calculation returned a result.');

    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );
    t.true(
      activityTypes.includes('calculateSum'),
      `calculateSum should be scheduled even with invalid input, got: ${activityTypes.join(', ')}`
    );
  });
});

// --- Wire contract tests ---

test('Wire contract: prompt and tracing survive round trip through wire projection', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const provider = new RequestCapturingModelProvider();
  const worker = await createWorker({
    plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })],
  });

  let result: Awaited<ReturnType<typeof wireRoundTripWorkflow>>;
  await worker.runUntil(async () => {
    result = await executeWorkflow(wireRoundTripWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
  });

  // --- Request side (captured by activity-side model) ---
  const req = provider.lastRequest as any;
  t.truthy(req, 'Expected model to have received a request');

  // Prompt field with nested structure should survive the wire
  t.truthy(req?.prompt, 'Expected prompt field to survive round trip');
  t.is(req?.prompt?.promptId, 'pt_round_trip', `Expected promptId 'pt_round_trip', got: ${req?.prompt?.promptId}`);
  t.deepEqual(req?.prompt?.variables, { key: 'value', nested: { deep: true } });

  // Tracing field should survive (default is false when tracing is disabled in workflow)
  t.true('tracing' in req, 'Expected tracing field to be present in wire request');

  // __wireVersion is stripped by fromSerializedModelRequest before reaching the model
  t.false('__wireVersion' in req, '__wireVersion should be stripped before reaching the model');

  // --- Response side (returned from activity to workflow) ---
  t.is(result!.usageInputTokens, 10, `Expected usage.inputTokens=10, got: ${result!.usageInputTokens}`);
  t.is(result!.usageOutputTokens, 8, `Expected usage.outputTokens=8, got: ${result!.usageOutputTokens}`);
  t.is(result!.outputLength, 1, `Expected output array length=1, got: ${result!.outputLength}`);
  t.false(result!.hasWireVersion, '__wireVersion should be stripped from response by fromSerializedModelResponse');
});

// Stripping is a structural guarantee: toSerializedModelRequest uses additive projection
// (only copies listed fields), so unlisted fields like `signal` can never leak through.
// This integration test verifies the end-to-end absence on the activity-side model request.
test('Wire contract: signal is stripped from wire request', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const provider = new RequestCapturingModelProvider();
  const worker = await createWorker({
    plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(wireStrippingCheckWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'captured');
  });

  const req = provider.lastRequest as any;
  t.truthy(req, 'Expected model to have received a request');
  t.false('signal' in req, 'signal should be stripped from wire request (AbortSignal is not serializable)');
});

test('Wire contract: version mismatch throws non-retryable WireVersionMismatch error', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Should not reach')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(wireVersionMismatchWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result.errorType, 'WireVersionMismatch', `Expected WireVersionMismatch error type, got: ${result.errorType}`);
    t.true(
      result.errorMessage.includes('wire version mismatch'),
      `Expected descriptive message about version mismatch, got: ${result.errorMessage}`
    );
  });
});

test('Wire contract: SerializedModelRequest shape snapshot', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const actualKeys = await executeWorkflow(wireRequestSnapshotWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    const expectedKeys = [
      '__wireVersion',
      'conversationId',
      'handoffs',
      'input',
      'modelSettings',
      'outputType',
      'overridePromptModel',
      'previousResponseId',
      'prompt',
      'systemInstructions',
      'tools',
      'toolsExplicitlyProvided',
      'tracing',
    ];
    t.deepEqual(
      actualKeys,
      expectedKeys,
      `SerializedModelRequest shape changed — bump WIRE_VERSION and update this snapshot. Got: ${actualKeys.join(', ')}`
    );
    t.true(actualKeys.includes('__wireVersion'), 'Wire version key must be present');
    t.false(actualKeys.includes('signal'), 'signal must not be on wire (AbortSignal is not serializable)');
  });
});

test('Wire contract: SerializedModelResponse shape snapshot', async (t) => {
  const response = {
    usage: {
      requests: 1,
      inputTokens: 10,
      outputTokens: 5,
      totalTokens: 15,
      inputTokensDetails: [],
      outputTokensDetails: [],
    },
    output: [{ type: 'message', content: 'test' }],
    responseId: 'resp_123',
    providerData: { key: 'value' },
  } as any;

  const wire = toSerializedModelResponse(response);
  const actualKeys = Object.keys(wire).sort();

  const expectedKeys = ['__wireVersion', 'output', 'providerData', 'responseId', 'usage'];
  t.deepEqual(
    actualKeys,
    expectedKeys,
    `SerializedModelResponse shape changed — bump WIRE_VERSION and update this snapshot. Got: ${actualKeys.join(', ')}`
  );
  t.is(wire.__wireVersion, 1, 'Wire version should be 1');
});

// Upstream-drift detection: verifies that all fields we project onto the wire are JSON-safe.
// If upstream changes a field type from a JSON-safe primitive to a class/Date/Map, this test
// fails, signaling that WIRE_VERSION needs a bump and the projection needs updating.
test('Wire contract: upstream ModelRequest fields survive JSON round-trip (drift detection)', async (t) => {
  const sampleRequest = {
    systemInstructions: 'You are a helpful assistant.',
    input: [{ role: 'user', content: [{ type: 'input_text', text: 'Hello' }], providerData: {} }],
    modelSettings: { temperature: 0.7, maxTokens: 100, topP: 0.9 },
    tools: [{ type: 'function', name: 'get_weather', parameters: { type: 'object', properties: {} }, strict: true }],
    toolsExplicitlyProvided: true,
    outputType: { type: 'text' },
    handoffs: [{ toolName: 'transfer_to_agent', toolDescription: 'Transfer', strictJsonSchema: true }],
    prompt: { promptId: 'pt_drift', version: 'v1', variables: { city: 'NYC' } },
    previousResponseId: 'resp_prev_001',
    conversationId: 'conv_drift_001',
    tracing: false,
    overridePromptModel: false,
  };

  const roundTripped = JSON.parse(JSON.stringify(sampleRequest));
  t.deepEqual(
    roundTripped,
    sampleRequest,
    'All upstream ModelRequest field values must survive JSON round-trip. ' +
      'If this fails, upstream introduced a non-JSON-safe field — bump WIRE_VERSION and update the projection.'
  );
});

test('Wire contract: upstream ModelResponse fields survive JSON round-trip (drift detection)', async (t) => {
  const sampleResponse = {
    usage: {
      requests: 1,
      inputTokens: 42,
      outputTokens: 15,
      totalTokens: 57,
      inputTokensDetails: [{ cachedTokens: 10 }],
      outputTokensDetails: [{ reasoningTokens: 5 }],
    },
    output: [
      {
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [{ type: 'output_text', text: 'Hello!' }],
        id: 'msg_drift_001',
        providerData: { model: 'gpt-4o' },
      },
    ],
    responseId: 'resp_drift_001',
    providerData: { model: 'gpt-4o', latencyMs: 150 },
  } as any;

  const wire = toSerializedModelResponse(sampleResponse);
  const roundTripped = JSON.parse(JSON.stringify(wire));
  t.deepEqual(
    roundTripped,
    wire,
    'All SerializedModelResponse field values must survive JSON round-trip. ' +
      'If this fails, upstream introduced a non-JSON-safe field — bump WIRE_VERSION and update the projection.'
  );
});

// --- Tracing span capture ---

test('OpenAI Agents tracing path is active and produces trace/span events', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Traced response')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(tracingSpanCaptureWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.true(result.traceIds.length > 0, 'Should capture at least one trace');
    t.true(result.spanTypes.includes('agent'), 'Should have an agent span');
    t.true(
      result.spanTypes.includes('generation') || result.spanTypes.includes('response'),
      'Should have a generation or response span'
    );
  });
});

// --- Replay-safety test ---

test('Tracing is replay-safe — no NondeterminismError when workflow replays', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Replayed response')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(replaySafetyWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.true(
      result.replayDetected,
      'Workflow should have detected replay (proves maxCachedWorkflows: 0 forced a replay)'
    );
    t.true(result.traceIds.length > 0, 'Should capture at least one trace during non-replay execution');
    t.true(result.spanTypes.includes('agent'), 'Should have an agent span');
  });
});

// --- Handoff-clone snapshot test ---

test('Handoff clone preserves all public fields through convertAgent', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(handoffCloneSnapshotWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    for (const [field, preserved] of Object.entries(result.fieldsPreserved)) {
      t.true(preserved, `Handoff clone field '${field}' should be preserved`);
    }
    t.true(result.agentReplaced, 'Handoff clone agent should be replaced with converted agent');
    t.true(result.onInvokeHandoffReplaced, 'Handoff clone onInvokeHandoff should be replaced with wrapper');
    t.true(result.prototypeMatch, 'Handoff clone prototype should match original');
  });
});

// --- Concurrent-workflow tracing isolation test ---

test('Concurrent workflows on same worker have isolated trace spans', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
          textResponse('Isolated response 1'),
          textResponse('Isolated response 2'),
        ]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const [result1, result2] = await Promise.all([
      executeWorkflow(concurrentTracingIsolationWorkflow, {
        workflowId: 'isolation-wf-1',
        workflowExecutionTimeout: '30 seconds',
      }),
      executeWorkflow(concurrentTracingIsolationWorkflow, {
        workflowId: 'isolation-wf-2',
        workflowExecutionTimeout: '30 seconds',
      }),
    ]);

    // Each workflow reports its own ID
    t.is(result1.workflowId, 'isolation-wf-1');
    t.is(result2.workflowId, 'isolation-wf-2');

    // Both workflows captured traces
    t.true(result1.traceIds.length > 0, 'Workflow 1 should capture at least one trace');
    t.true(result2.traceIds.length > 0, 'Workflow 2 should capture at least one trace');

    // Both workflows captured spans
    t.true(result1.spanTypes.includes('agent'), 'Workflow 1 should have an agent span');
    t.true(result2.spanTypes.includes('agent'), 'Workflow 2 should have an agent span');

    // No cross-pollination: trace IDs should be disjoint between the two workflows
    const sharedTraces = result1.traceIds.filter((id) => result2.traceIds.includes(id));
    t.is(sharedTraces.length, 0, 'No shared trace IDs between concurrent workflows');
  });
});

// --- Trace context propagation across workflow/activity boundary ---

test('Agent trace context propagates across workflow/activity boundary', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new TraceCaptureModelProvider(),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(traceContextPropagationWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.truthy(result.workflowTraceId, 'Workflow should have a traceId');
    t.truthy(result.activityTraceId, 'Activity should capture a traceId');
    t.not(result.activityTraceId, 'NO_TRACE', 'Activity should have propagated trace context');
    t.is(
      result.activityTraceId,
      result.workflowTraceId,
      'Activity-side traceId must match workflow-side traceId (proves propagation)'
    );
  });
});

// --- Client→workflow trace context propagation ---

test('Client-side trace context propagates to workflow via interceptor', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('x')]),
      }),
    ],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [new OpenAIAgentsTraceClientInterceptor()],
  });

  await worker.runUntil(async () => {
    const result = await withTrace('client-test-trace', async (trace) => {
      const clientTraceId = trace.traceId;
      const workflowTraceId = await wfClient.execute(clientToWorkflowTraceWorkflow, {
        taskQueue,
        workflowId: `t5-client-trace-${Date.now()}`,
        workflowExecutionTimeout: '30 seconds',
      });
      return { clientTraceId, workflowTraceId };
    });

    t.truthy(result.clientTraceId, 'Client should have a trace ID');
    t.not(result.workflowTraceId, 'NO_TRACE', 'Workflow should have restored trace context');
    t.is(
      result.workflowTraceId,
      result.clientTraceId,
      'Workflow-side traceId must match client-side traceId (proves client→workflow propagation)'
    );
  });
});

// --- Signal trace context propagation ---

test('Signal carries trace context across workflow boundary', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('x')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(signalTracePropagationParentWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.truthy(result.parentTraceId, 'Parent should have a trace ID');
    t.not(result.signalTraceId, 'NO_SIGNAL_TRACE', 'Signal handler should have restored trace context');
    t.is(
      result.signalTraceId,
      result.parentTraceId,
      'Signal handler traceId must match parent traceId (proves signal propagation)'
    );
  });
});

// --- Child workflow trace context propagation ---

test('Child workflow receives propagated trace context from parent', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('x')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(childWorkflowTracePropagationParentWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.truthy(result.parentTraceId, 'Parent should have a trace ID');
    t.not(result.childTraceId, 'NO_TRACE', 'Child should have restored trace context');
    t.is(
      result.childTraceId,
      result.parentTraceId,
      'Child traceId must match parent traceId (proves child workflow propagation)'
    );
  });
});

// --- Deterministic trace/span IDs and timestamps ---

test('Trace/span IDs and timestamps are deterministic across replay', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Deterministic IDs')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(deterministicTraceIdsWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    // --- ID format verification ---
    t.true(result.traceIds.length > 0, 'Should capture at least one trace ID');
    t.true(result.spanIds.length > 0, 'Should capture at least one span ID');

    for (const traceId of result.traceIds) {
      t.regex(traceId, /^trace_[0-9a-f]{32}$/, `Trace ID '${traceId}' should match deterministic format trace_<32hex>`);
    }

    for (const spanId of result.spanIds) {
      t.regex(spanId, /^span_[0-9a-f]{24}$/, `Span ID '${spanId}' should match deterministic format span_<24hex>`);
    }

    // --- Timestamp determinism verification ---
    // The workflow runs with maxCachedWorkflows: 0, which forces replay.
    // If Date / new Date() were non-deterministic in the sandbox, the replayed
    // command sequence would diverge → NondeterminismError. Reaching this point
    // proves the sandbox clock is deterministic. We additionally verify the
    // captured values are valid ISO 8601 timestamps.
    t.regex(
      result.workflowTimestamp,
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/,
      'Workflow Date().toISOString() should produce valid ISO 8601 timestamp'
    );

    for (const ts of result.spanStartTimestamps) {
      t.regex(
        ts,
        /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/,
        `Span startedAt '${ts}' should be a valid ISO 8601 timestamp (from timeIso())`
      );
    }
  });
});

// --- Stateful MCP ---

test('Stateful MCP: not connected produces ApplicationFailure', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(statefulMcpNotConnectedWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'Stateful MCP Server not connected. Call connect first.');
  });
});

test('Stateful MCP: no dedicated worker produces DedicatedWorkerFailure', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  // Override the session activity to just sleep (no dedicated worker started)
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
    activities: {
      // Override: session activity that does nothing useful — no dedicated worker is started.
      // The workflow's listTools call will time out on scheduleToStart because
      // no worker is polling the per-run task queue.
      'testStateful-stateful-server-session': async () => {
        await new Promise((resolve) => setTimeout(resolve, 30_000));
      },
    },
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpNoWorkerWorkflow, {
      args: [1000],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'DedicatedWorkerFailure: MCP Stateful Server Worker failed to schedule activity.');
  });
});

test('Stateful MCP: happy path with agent', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const mcpProvider = new StatefulMCPServerProvider(
    'testStateful',
    () => ({
      async connect() {},
      async cleanup() {},
      async listTools() {
        return [
          {
            name: 'get_status',
            description: 'Returns status',
            inputSchema: { type: 'object' as const, properties: {}, required: [] as string[], additionalProperties: false },
          },
        ];
      },
      async callTool() {
        return [{ type: 'text', text: 'status-ok' }];
      },
    }),
    t.context.env.nativeConnection
  );

  function* statefulMcpGenerator() {
    yield toolCallResponse('get_status', {});
    yield textResponse('Status is ok.');
  }

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => statefulMcpGenerator()),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpAgentWorkflow, {
      args: ['Check the status'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Status is ok.');
  });
});

test('Stateful MCP: multi-run isolation under reuseV8Context', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  let callCount = 0;
  const mcpProvider = new StatefulMCPServerProvider(
    'isolationTest',
    () => {
      // Each factory call produces a server with a unique marker
      const marker = `server-${++callCount}`;
      return {
        async connect() {},
        async cleanup() {},
        async listTools() {
          return [
            {
              name: 'get_marker',
              description: 'Returns marker',
              inputSchema: { type: 'object' as const, properties: {}, required: [] as string[], additionalProperties: false },
            },
          ];
        },
        async callTool() {
          return [{ type: 'text', text: marker }];
        },
      };
    },
    t.context.env.nativeConnection
  );

  const worker = await createWorker({
    reuseV8Context: true,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const resultA = await executeWorkflow(statefulMcpIsolationWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });
    const resultB = await executeWorkflow(statefulMcpIsolationWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    // Each workflow must have gotten its own server instance
    t.not(resultA, resultB, 'Two runs on the same V8 isolate must see distinct server instances');
    t.regex(resultA, /server-1/, 'First run should see server-1');
    t.regex(resultB, /server-2/, 'Second run should see server-2');
  });
});

test('Stateful MCP: heartbeat timeout produces DedicatedWorkerFailure', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const mcpProvider = new StatefulMCPServerProvider(
    'heartbeatTest',
    () => ({
      async connect() {},
      async cleanup() {},
      async listTools(): Promise<any[]> {
        // Block long enough for the 1-second heartbeat timeout to fire
        await new Promise((resolve) => setTimeout(resolve, 10_000));
        return [];
      },
      async callTool() {
        return [];
      },
    }),
    t.context.env.nativeConnection
  );

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpHeartbeatTimeoutWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'DedicatedWorkerFailure: MCP Stateful Server Worker failed to heartbeat.');
  });
});

test('Stateful MCP: slow connect heartbeat regression', async (t) => {
  // Regression test (Item 31): the session activity heartbeats BEFORE
  // server.connect() returns. Without the immediate heartbeat() call,
  // a slow connect() would timeout because setInterval(heartbeat, 30_000)
  // only fires at t=30s — too late for a short heartbeatTimeout.
  const { createWorker, startWorkflow } = helpers(t);

  let connectCallCount = 0;
  const mcpProvider = new StatefulMCPServerProvider(
    'slowConnectTest',
    () => ({
      async connect() {
        connectCallCount++;
        // Simulate a slow connection — 1.5 seconds
        await new Promise((resolve) => setTimeout(resolve, 1500));
      },
      async cleanup() {},
      async listTools() {
        return [
          {
            name: 'dummy',
            description: 'Dummy tool',
            inputSchema: { type: 'object' as const, properties: {}, required: [] as string[], additionalProperties: false },
          },
        ];
      },
      async callTool() {
        return [];
      },
    }),
    t.context.env.nativeConnection
  );

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpSlowConnectHeartbeatWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    // The workflow should succeed — the immediate heartbeat prevents
    // the session activity from timing out during the slow connect.
    t.regex(result, /^connected:/, `Expected connected:N result, got: ${result}`);
    t.true(connectCallCount > 0, 'connect() should have been called');
  });
});

test('Stateful MCP: replay safety with maxCachedWorkflows 0', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const mcpProvider = new StatefulMCPServerProvider(
    'replayTest',
    () => ({
      async connect() {},
      async cleanup() {},
      async listTools() {
        return [
          {
            name: 'get_data',
            description: 'Returns data',
            inputSchema: { type: 'object' as const, properties: {}, required: [] as string[], additionalProperties: false },
          },
        ];
      },
      async callTool() {
        return [{ type: 'text', text: 'replay-safe-data' }];
      },
    }),
    t.context.env.nativeConnection
  );

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(statefulMcpReplayWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    // If replay caused NondeterminismError, the workflow would fail instead of returning
    t.regex(result, /replay-safe-data/, 'Workflow must complete without NondeterminismError');
  });
});

// --- Query trace context propagation ---

test('Query carries trace context from client to workflow', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('x')]),
      }),
    ],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [new OpenAIAgentsTraceClientInterceptor()],
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(queryTracePropagationWorkflow, {
      taskQueue,
      workflowId: `query-trace-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await withTrace('query-trace-test', async (trace) => {
      const clientTraceId = trace.traceId;

      // Poll until the query handler is registered
      let queryTraceId = 'NO_QUERY_TRACE';
      for (let i = 0; i < 30; i++) {
        try {
          queryTraceId = await handle.query<string>('queryTraceId');
          break;
        } catch {
          await new Promise((r) => setTimeout(r, 500));
        }
      }

      return { clientTraceId, queryTraceId };
    });

    // Signal the workflow to complete
    await handle.signal('queryTraceTestDone');
    await handle.result();

    t.truthy(result.clientTraceId, 'Client should have a trace ID');
    t.not(result.queryTraceId, 'NO_QUERY_TRACE', 'Query handler should have restored trace context');
    t.is(
      result.queryTraceId,
      result.clientTraceId,
      'Query handler traceId must match client-side traceId (proves query propagation)'
    );
  });
});

// --- Config header propagation tests (H1) ---

test('Config header propagated via startWithDetails', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentsPlugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('x')]),
    interceptorOptions: { addTemporalSpans: true },
    modelParams: { taskQueue: 'plugin-tq' },
  });

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [agentsPlugin],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        addTemporalSpans: true,
        modelParams: { taskQueue: 'plugin-tq' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await wfClient.execute(configPropagationWorkflow, {
      taskQueue,
      workflowId: `config-start-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result.addTemporalSpans, true, 'addTemporalSpans should be propagated from plugin');
    t.is(result.taskQueue, 'plugin-tq', 'modelParams.taskQueue should be propagated from plugin');
  });
});

test('Config header propagated via startWithDetails (plugin auto-wired)', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentsPlugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('x')]),
    interceptorOptions: { addTemporalSpans: true },
    modelParams: { taskQueue: 'plugin-tq' },
  });

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [agentsPlugin],
  });

  // Use Client with plugins — the plugin's configureClient auto-wires
  // the OpenAIAgentsTraceClientInterceptor onto the Client's workflow
  // interceptors, exercising the auto-wiring contract.
  const client = new Client({
    connection: (t.context as any).env.connection,
    plugins: [agentsPlugin],
  });

  await worker.runUntil(async () => {
    const result = await client.workflow.execute(configPropagationWorkflow, {
      taskQueue,
      workflowId: `config-start-auto-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result.addTemporalSpans, true, 'addTemporalSpans should be propagated via plugin auto-wiring');
    t.is(result.taskQueue, 'plugin-tq', 'modelParams.taskQueue should be propagated via plugin auto-wiring');
  });
});

test('Config header propagated via signalWithStart', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentsPlugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('x')]),
    interceptorOptions: { addTemporalSpans: true },
    modelParams: { taskQueue: 'plugin-tq' },
  });

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [agentsPlugin],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        addTemporalSpans: true,
        modelParams: { taskQueue: 'plugin-tq' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    // signalWithStart starts the workflow AND sends a signal atomically
    const handle = await wfClient.signalWithStart(configPropagationWorkflow, {
      taskQueue,
      workflowId: `config-signal-start-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
      signal: 'unused-signal',
      signalArgs: [],
    });

    const result = await handle.result();
    t.is(result.addTemporalSpans, true, 'addTemporalSpans should be propagated via signalWithStart');
    t.is(result.taskQueue, 'plugin-tq', 'modelParams.taskQueue should be propagated via signalWithStart');
  });
});

test('Config header propagated via startUpdateWithStart', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentsPlugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('x')]),
    interceptorOptions: { addTemporalSpans: true },
    modelParams: { taskQueue: 'plugin-tq' },
  });

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [agentsPlugin],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        addTemporalSpans: true,
        modelParams: { taskQueue: 'plugin-tq' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    // executeUpdateWithStart starts the workflow AND sends an update atomically.
    // The workflow has an update handler that returns the observed config.
    const startOp = new WithStartWorkflowOperation(configUpdateWithStartWorkflow, {
      taskQueue,
      workflowId: `config-update-start-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
      workflowIdConflictPolicy: WorkflowIdConflictPolicy.FAIL,
    });

    const updateResult = await wfClient.executeUpdateWithStart('configUpdateWithStart', {
      startWorkflowOperation: startOp,
    });

    t.is(
      (updateResult as any).addTemporalSpans,
      true,
      'addTemporalSpans should be propagated via startUpdateWithStart'
    );
    t.is(
      (updateResult as any).taskQueue,
      'plugin-tq',
      'modelParams.taskQueue should be propagated via startUpdateWithStart'
    );
  });
});

test('Config header propagated via continueAsNew', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentsPlugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('x')]),
    interceptorOptions: { addTemporalSpans: true },
    modelParams: { taskQueue: 'plugin-tq' },
  });

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [agentsPlugin],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        addTemporalSpans: true,
        modelParams: { taskQueue: 'plugin-tq' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await wfClient.execute(configContinueAsNewWorkflow, {
      taskQueue,
      workflowId: `config-continue-as-new-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
      args: [0], // iteration=0, will continueAsNew to iteration=1
    });

    t.is(result.addTemporalSpans, true, 'addTemporalSpans should survive continueAsNew');
    t.is(result.taskQueue, 'plugin-tq', 'modelParams.taskQueue should survive continueAsNew');
  });
});

test('Config header propagated to child workflows', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentsPlugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('x')]),
    interceptorOptions: { addTemporalSpans: true },
    modelParams: { taskQueue: 'plugin-tq' },
  });

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [agentsPlugin],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        addTemporalSpans: true,
        modelParams: { taskQueue: 'plugin-tq' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await wfClient.execute(configChildParentWorkflow, {
      taskQueue,
      workflowId: `config-child-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result.addTemporalSpans, true, 'addTemporalSpans should be propagated to child workflow');
    t.is(result.taskQueue, 'plugin-tq', 'modelParams.taskQueue should be propagated to child workflow');
  });
});

// --- Override precedence test (H1) ---

test('Runner constructor args override plugin config header values', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentsPlugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('x')]),
    interceptorOptions: { addTemporalSpans: true },
    modelParams: { taskQueue: 'plugin-tq' },
  });

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [agentsPlugin],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        addTemporalSpans: true,
        modelParams: { taskQueue: 'plugin-tq' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await wfClient.execute(configOverridePrecedenceWorkflow, {
      taskQueue,
      workflowId: `config-override-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
    });

    // Runner overrides addTemporalSpans=false (plugin had true)
    t.is(result.addTemporalSpans, false, 'Runner addTemporalSpans=false should override plugin true');
    // Runner overrides taskQueue='runner-override' (plugin had 'plugin-tq')
    t.is(result.taskQueue, 'runner-override', 'Runner taskQueue should override plugin taskQueue');
  });
});

// --- Fallback test (H1): no plugin client interceptor ---

test('Runner populates store from own args when no plugin client interceptor is wired', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('x')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    // env.client.workflow.execute does NOT have the plugin client interceptor,
    // so no __openai_agents_config header is injected. The workflow's runner
    // constructor should still populate the store from its own args.
    const result = await executeWorkflow(configFallbackWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(
      result.addTemporalSpans,
      true,
      'Runner args should populate store even without plugin client interceptor'
    );
  });
});

// --- H3 regression: modelParams isolation between runners ---

test('modelParams override does not leak between runners in the same workflow', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('x')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(configModelParamsIsolationWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.is(result.r1TaskQueue, 'a', 'Runner 1 should have taskQueue=a');
    // Runner 2 did NOT set taskQueue — it should NOT inherit 'a' from runner 1
    t.not(
      result.r2TaskQueue,
      'a',
      'Runner 2 should NOT inherit taskQueue from runner 1 (accumulation bug)'
    );
    t.is(result.r2StartToCloseTimeout, '5s', 'Runner 2 should have its own startToCloseTimeout');
  });
});

// --- M1 regression: summaryOverride function form stripped before child workflow propagation ---

// **Discrimination design (rock-solid):**
// Pre-M1, injectConfigHeaderFromStore forwarded config.modelParams directly into
// the wire header. JSON serialization would silently mangle the function form
// (drop the function method, leaving {} for the parent object or stripping the
// field entirely depending on JS spec corner cases).
//
// Post-M1, the function explicitly narrows summaryOverride (drops non-string
// values via destructure + selective reattach) BEFORE injecting into the wire
// header. The test discriminates by checking BOTH:
//   1. summaryOverride is undefined (stripped — non-string form removed)
//   2. taskQueue === 'sibling-task-queue' (sibling preserved intact)
//
// The sibling assertion proves the strip is targeted — not collateral damage
// from a broken serialization path. Without the M1 narrowing, a future bug that
// drops siblings while stripping summaryOverride would be caught.
test('summaryOverride function form set via runner is stripped before child workflow propagation', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentsPlugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('x')]),
  });

  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [agentsPlugin],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [new OpenAIAgentsTraceClientInterceptor()],
  });

  await worker.runUntil(async () => {
    const result = await wfClient.execute(summaryOverrideFunctionStripParentWorkflow, {
      taskQueue,
      workflowId: `summary-fn-strip-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds',
    });

    // The parent set summaryOverride to a function-form (ModelSummaryProvider).
    // injectConfigHeaderFromStore must strip it — functions can't survive JSON.
    // The child should see undefined, not a corrupted shape like {} or "[Function]".
    t.is(
      result.summaryOverride,
      undefined,
      'Function-form summaryOverride must be stripped before child workflow propagation'
    );

    // Sibling field must survive — proves the M1 destructure-and-reattach in
    // injectConfigHeaderFromStore correctly preserves serializable siblings
    // while stripping non-serializable summaryOverride.
    t.is(
      result.taskQueue,
      'sibling-task-queue',
      'Sibling modelParams fields must propagate intact through injectConfigHeaderFromStore'
    );
  });
});
