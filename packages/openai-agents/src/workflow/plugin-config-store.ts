/**
 * Per-workflow plugin configuration store.
 *
 * Under `reuseV8Context: true`, multiple workflow executions share a single
 * V8 isolate. Any per-workflow state MUST be keyed by `workflowInfo().workflowId`
 * — module-level singletons, plain `globalThis` writes, and untyped maps will
 * silently leak state between workflows in the same isolate.
 *
 * Lifecycle:
 * - **Set (original)** by the workflow inbound interceptor on `execute` via
 *   {@link setOriginalPluginConfig}. Written once per workflow execution from
 *   the decoded `__openai_agents_config` header. Never overwritten by runners.
 * - **Set (effective)** by the `TemporalOpenAIRunner` constructor via
 *   {@link setPluginConfig}. Each runner construction merges its own args
 *   with the original config (not the previous effective config), then writes
 *   the result here.
 * - **Read** by the workflow trace interceptor ({@link getCurrentPluginConfig})
 *   and the runner ({@link getOriginalPluginConfig}).
 * - **Cleared** by the workflow inbound interceptor on workflow completion
 *   (try/finally around `next(input)` in `execute`) to prevent unbounded growth.
 *
 * The multi-workflow isolation invariant is guarded by the test
 * "multi-workflow config isolation under reuseV8Context" in
 * `test-openai-agents-tracing.ts`.
 */
import { workflowInfo } from '@temporalio/workflow';
import type { ModelActivityOptions } from '../common/model-activity-options';

export interface PluginConfig {
  addTemporalSpans: boolean;
  startTraces: boolean;
  modelParams: ModelActivityOptions;
}

const configByWorkflowId = new Map<string, PluginConfig>();

/**
 * Original (header-derived) config, written once by the inbound interceptor.
 * Runner constructors read from this map for merge defaults, ensuring that
 * constructing multiple runners in the same workflow does not accumulate
 * overrides from earlier runners.
 */
const originalConfigByWorkflowId = new Map<string, PluginConfig>();

/** Write the effective config for a workflow. Called by the runner constructor. */
export function setPluginConfig(workflowId: string, config: PluginConfig): void {
  configByWorkflowId.set(workflowId, config);
}

/** Read the effective config for a workflow. */
export function getPluginConfig(workflowId: string): PluginConfig | undefined {
  return configByWorkflowId.get(workflowId);
}

/**
 * Write the original header-derived config for a workflow.
 * Called once by the inbound interceptor on `execute`. Runners never write here.
 */
export function setOriginalPluginConfig(workflowId: string, config: PluginConfig): void {
  originalConfigByWorkflowId.set(workflowId, config);
}

/**
 * Read the original header-derived config for a workflow.
 * Used by the runner constructor as the merge base, so that multiple runner
 * constructions don't accumulate each other's overrides.
 */
export function getOriginalPluginConfig(workflowId: string): PluginConfig | undefined {
  return originalConfigByWorkflowId.get(workflowId);
}

export function clearPluginConfig(workflowId: string): void {
  configByWorkflowId.delete(workflowId);
  originalConfigByWorkflowId.delete(workflowId);
}

/** Convenience — reads the effective config for the currently-executing workflow. */
export function getCurrentPluginConfig(): PluginConfig | undefined {
  return configByWorkflowId.get(workflowInfo().workflowId);
}
