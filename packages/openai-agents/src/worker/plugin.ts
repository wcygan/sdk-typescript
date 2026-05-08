import type { ModelProvider } from '@openai/agents-core';
import { SimplePlugin } from '@temporalio/plugin';
import { OpenAIAgentsTraceClientInterceptor } from '../client/trace-interceptor';
import type { SerializableModelActivityOptions } from '../common/model-activity-options';
import { createModelActivity } from './activities';
import type { StatelessMCPServerProvider } from './mcp-provider';
import type { StatefulMCPServerProvider } from './stateful-mcp-provider';
import {
  OpenAIAgentsTraceActivityInboundInterceptor,
  type OpenAIAgentsTraceInterceptorOptions,
} from './trace-interceptor';

/** Either a stateless or stateful MCP server provider. */
export type MCPServerProvider = StatelessMCPServerProvider | StatefulMCPServerProvider;

/**
 * Options controlling trace interceptor behavior on both the activity side
 * (directly) and the workflow side (via header propagation).
 */
export interface OpenAIAgentsPluginInterceptorOptions {
  /** Wrap calls in `temporal:*` custom spans. @default false */
  addTemporalSpans?: boolean;
  /** Fire processor events on restored trace contexts. @default false */
  startTraces?: boolean;
}

/**
 * Options for the OpenAI Agents plugin.
 */
export interface OpenAIAgentsPluginOptions {
  /** The model provider to use for resolving model names to Model instances (e.g. OpenAIProvider) */
  modelProvider: ModelProvider;
  /**
   * MCP server providers whose activities will be auto-registered.
   * Accepts both stateless and stateful providers.
   */
  mcpServerProviders?: MCPServerProvider[];
  /**
   * Default model activity options (timeouts, retry, task queue, etc.).
   *
   * Propagated to the workflow via the `__openai_agents_config` header
   * when the plugin's client interceptor is wired. The runner's
   * {@link TemporalOpenAIRunnerOptions.modelParams} override these per-field.
   *
   * Typed as {@link SerializableModelActivityOptions} — the function form of
   * `summaryOverride` (`ModelSummaryProvider`) is excluded at compile time
   * because it cannot survive JSON serialization through the config header.
   * Pass function-form overrides via the runner constructor in workflow code.
   */
  modelParams?: SerializableModelActivityOptions;
  /**
   * Options controlling trace interceptor behavior (temporal span wrapping
   * and trace event firing).
   *
   * Propagated to the workflow via the `__openai_agents_config` header
   * when the plugin's client interceptor is wired. The runner's
   * `addTemporalSpans` / `startTraces` override these per-field.
   *
   * Also configures the activity-side interceptor directly.
   */
  interceptorOptions?: OpenAIAgentsPluginInterceptorOptions;
}

/**
 * A Temporal plugin that integrates the OpenAI Agents SDK for use in workflows.
 * Registers model invocation activities so that workflow-side ActivityBackedModel
 * can delegate LLM calls to the activity worker.
 */
export class OpenAIAgentsPlugin extends SimplePlugin {
  constructor(options: OpenAIAgentsPluginOptions) {
    const modelActivities = createModelActivity(options.modelProvider);

    let allActivities: Record<string, (...args: any[]) => Promise<any>> = { ...modelActivities };

    if (options.mcpServerProviders) {
      const seenNames = new Set<string>();
      for (const provider of options.mcpServerProviders) {
        if (seenNames.has(provider.name)) {
          throw new Error(
            `Duplicate MCP server provider name: '${provider.name}'. Each provider must have a unique name — activity keys collide.`
          );
        }
        seenNames.add(provider.name);
        const providerActivities = provider._getActivities();
        allActivities = { ...allActivities, ...providerActivities };
      }
    }

    const interceptorOpts = options.interceptorOptions;
    const activityInterceptorOptions: OpenAIAgentsTraceInterceptorOptions = {
      addTemporalSpans: interceptorOpts?.addTemporalSpans,
      startTraces: interceptorOpts?.startTraces,
    };

    super({
      name: 'OpenAIAgentsPlugin',
      activities: allActivities,
      clientInterceptors: {
        workflow: [
          new OpenAIAgentsTraceClientInterceptor({
            addTemporalSpans: interceptorOpts?.addTemporalSpans,
            startTraces: interceptorOpts?.startTraces,
            modelParams: options.modelParams,
          }),
        ],
      },
      workerInterceptors: {
        workflowModules: [require.resolve('../workflow/trace-interceptor')],
        activity: [
          (ctx) => ({ inbound: new OpenAIAgentsTraceActivityInboundInterceptor(ctx, activityInterceptorOptions) }),
        ],
      },
    });
  }
}
