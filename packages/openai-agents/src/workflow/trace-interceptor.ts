import { getCurrentTrace, getCurrentSpan, withCustomSpan } from '@openai/agents-core';
import type {
  ActivityInput,
  LocalActivityInput,
  Next,
  StartChildWorkflowExecutionInput,
  SignalWorkflowInput,
  WorkflowInterceptors,
  WorkflowOutboundCallsInterceptor,
} from '@temporalio/workflow';
import { type AgentsSpanHeader, injectAgentsTraceHeader } from '../common/trace-header';

function currentAgentsSpanHeader(): AgentsSpanHeader | null {
  const trace = getCurrentTrace();
  if (!trace) return null;
  const span = getCurrentSpan();
  return {
    traceName: trace.name ?? 'Unknown Workflow',
    spanId: span?.spanId ?? null,
    traceId: trace.traceId,
  };
}

/**
 * Workflow outbound interceptor that injects the active OpenAI Agents
 * trace/span context into outbound activity, child workflow, and signal
 * headers under the `__openai_span` key. Combined with
 * `OpenAIAgentsTraceActivityInboundInterceptor`, this propagates the
 * agent's trace tree across the workflow→activity boundary.
 */
export class OpenAIAgentsTraceOutboundInterceptor implements WorkflowOutboundCallsInterceptor {
  async scheduleActivity(
    input: ActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleActivity'>
  ): Promise<unknown> {
    const header = currentAgentsSpanHeader();
    if (!header) return next(input);

    const headers = injectAgentsTraceHeader(input.headers, header);
    return withCustomSpan(() => next({ ...input, headers }), {
      data: { name: `temporal:startActivity:${input.activityType}`, data: { activityType: input.activityType } },
    });
  }

  async scheduleLocalActivity(
    input: LocalActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleLocalActivity'>
  ): Promise<unknown> {
    const header = currentAgentsSpanHeader();
    if (!header) return next(input);

    const headers = injectAgentsTraceHeader(input.headers, header);
    return withCustomSpan(() => next({ ...input, headers }), {
      data: { name: `temporal:startLocalActivity:${input.activityType}`, data: { activityType: input.activityType } },
    });
  }

  async startChildWorkflowExecution(
    input: StartChildWorkflowExecutionInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'startChildWorkflowExecution'>
  ): Promise<[Promise<string>, Promise<unknown>]> {
    const header = currentAgentsSpanHeader();
    if (!header) return next(input);

    const headers = injectAgentsTraceHeader(input.headers, header);
    return next({ ...input, headers });
  }

  async signalWorkflow(
    input: SignalWorkflowInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'signalWorkflow'>
  ): Promise<void> {
    const header = currentAgentsSpanHeader();
    if (!header) return next(input);

    const headers = injectAgentsTraceHeader(input.headers, header);
    return next({ ...input, headers });
  }
}

export const interceptors = (): WorkflowInterceptors => ({
  outbound: [new OpenAIAgentsTraceOutboundInterceptor()],
});
