import {
  ActivityFailure,
  ApplicationFailure,
  TimeoutFailure,
  TimeoutType,
  type ActivityOptions,
  type Duration,
  type RetryPolicy,
} from '@temporalio/common';
import { CancellationScope, isCancellation, scheduleActivity, workflowInfo } from '@temporalio/workflow';
import type { TemporalMCPServer, MCPPromptDefinition } from './mcp-client';

export interface StatefulMcpServerOptions {
  /** Activity options for tool/prompt operation activities on the dedicated worker. */
  config?: {
    startToCloseTimeout?: Duration;
    scheduleToStartTimeout?: Duration;
    heartbeatTimeout?: Duration;
    taskQueue?: string;
    retryPolicy?: RetryPolicy;
  };
  /** Activity options for the long-running server-session activity. */
  serverSessionConfig?: {
    startToCloseTimeout?: Duration;
    heartbeatTimeout?: Duration;
  };
  /** Optional argument passed to the server factory on the worker side. */
  factoryArgument?: unknown;
}

/**
 * Cross-SDK contract: error type string thrown when the dedicated MCP worker
 * fails to schedule an activity or misses a heartbeat. Callers should catch
 * `ApplicationFailure` and check `failure.type === DEDICATED_WORKER_FAILURE_TYPE`.
 * Stable across SDK versions; do not rename.
 */
export const DEDICATED_WORKER_FAILURE_TYPE = 'DedicatedWorkerFailure';

/** Public contract — do not change wording. */
export const DEDICATED_WORKER_SCHEDULE_FAILURE_MESSAGE = 'MCP Stateful Server Worker failed to schedule activity.';

/** Public contract — do not change wording. */
export const DEDICATED_WORKER_HEARTBEAT_FAILURE_MESSAGE = 'MCP Stateful Server Worker failed to heartbeat.';

/**
 * Wraps an activity call so that schedule-to-start and heartbeat timeouts are
 * surfaced as `ApplicationFailure` with type `DedicatedWorkerFailure`.
 */
async function handleWorkerFailure<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err: unknown) {
    if (err instanceof ActivityFailure && err.cause instanceof TimeoutFailure) {
      if (err.cause.timeoutType === TimeoutType.SCHEDULE_TO_START) {
        throw ApplicationFailure.create({
          message: DEDICATED_WORKER_SCHEDULE_FAILURE_MESSAGE,
          type: DEDICATED_WORKER_FAILURE_TYPE,
          nonRetryable: false,
        });
      }
      if (err.cause.timeoutType === TimeoutType.HEARTBEAT) {
        throw ApplicationFailure.create({
          message: DEDICATED_WORKER_HEARTBEAT_FAILURE_MESSAGE,
          type: DEDICATED_WORKER_FAILURE_TYPE,
          nonRetryable: false,
        });
      }
    }
    throw err;
  }
}

/**
 * Workflow-side handle for a stateful MCP server connection.
 *
 * Maintains a persistent MCP server connection via a dedicated in-process
 * worker on a per-run task queue. Tool and prompt operations are routed to
 * that dedicated worker, preserving server state across calls.
 *
 * Usage:
 * ```ts
 * const server = statefulMcpServer('my-server');
 * await server.connect();
 * try {
 *   // use server.listTools(), server.callTool(), etc.
 * } finally {
 *   await server.cleanup();
 * }
 * ```
 */
export class StatefulMCPServerReference implements TemporalMCPServer {
  private readonly _name: string;
  private readonly _operationConfig: ActivityOptions;
  private readonly _sessionConfig: ActivityOptions;
  private readonly _factoryArgument: unknown;
  private _sessionScope: CancellationScope | undefined;
  private _sessionPromise: Promise<void> | undefined;
  private _connected = false;

  constructor(name: string, options?: StatefulMcpServerOptions) {
    this._name = `${name}-stateful`;
    this._operationConfig = {
      startToCloseTimeout: options?.config?.startToCloseTimeout ?? '1 minute',
      scheduleToStartTimeout: options?.config?.scheduleToStartTimeout ?? '30 seconds',
      heartbeatTimeout: options?.config?.heartbeatTimeout,
      taskQueue: options?.config?.taskQueue,
      retry: options?.config?.retryPolicy,
    };
    this._sessionConfig = {
      startToCloseTimeout: options?.serverSessionConfig?.startToCloseTimeout ?? '1 hour',
      heartbeatTimeout: options?.serverSessionConfig?.heartbeatTimeout,
    };
    this._factoryArgument = options?.factoryArgument;
  }

  get name(): string {
    return this._name;
  }

  get cacheToolsList(): boolean {
    return false;
  }

  /**
   * Starts the server-session activity on a per-run dedicated task queue.
   * Subsequent tool/prompt operations route to that task queue.
   */
  async connect(): Promise<void> {
    // Route operation activities to the dedicated per-run task queue
    const runId = workflowInfo().runId;
    this._operationConfig.taskQueue = `${this._name}@${runId}`;

    const sessionActivityName = `${this._name}-server-session`;

    // Start the long-running session activity in a dedicated cancellation scope
    this._sessionScope = new CancellationScope();
    this._sessionPromise = this._sessionScope.run(() =>
      scheduleActivity<void>(
        sessionActivityName,
        this._factoryArgument !== undefined ? [{ factoryArgument: this._factoryArgument }] : [undefined],
        this._sessionConfig
      )
    );
    // Don't block on the session — it runs until cancelled
    this._sessionPromise.catch(() => {
      // Swallowed here; errors are observed during cleanup
    });
    this._connected = true;
  }

  /**
   * Cancels the server-session activity, tearing down the dedicated worker.
   */
  async cleanup(): Promise<void> {
    if (this._sessionScope) {
      this._sessionScope.cancel();
      try {
        await this._sessionPromise;
      } catch (err: unknown) {
        if (!isCancellation(err)) {
          throw err;
        }
      }
      this._sessionScope = undefined;
      this._sessionPromise = undefined;
      this._connected = false;
    }
  }

  /**
   * Alias for `cleanup()` — matches the MCPServer interface.
   */
  async close(): Promise<void> {
    await this.cleanup();
  }

  private assertConnected(): void {
    if (!this._connected) {
      throw ApplicationFailure.create({
        message: 'Stateful MCP Server not connected. Call connect first.',
      });
    }
  }

  async listTools(): Promise<any[]> {
    this.assertConnected();
    return handleWorkerFailure(() =>
      scheduleActivity(this._name + '-list-tools', [], this._operationConfig)
    );
  }

  async callTool(toolName: string, args: Record<string, unknown> | null): Promise<any> {
    this.assertConnected();
    return handleWorkerFailure(() =>
      scheduleActivity(this._name + '-call-tool-v2', [{ toolName, args }], this._operationConfig)
    );
  }

  async listPrompts(): Promise<MCPPromptDefinition[]> {
    this.assertConnected();
    return handleWorkerFailure(() =>
      scheduleActivity(this._name + '-list-prompts', [], this._operationConfig)
    );
  }

  async getPrompt(
    promptName: string,
    args?: Record<string, unknown> | null
  ): Promise<unknown> {
    this.assertConnected();
    return handleWorkerFailure(() =>
      scheduleActivity(
        this._name + '-get-prompt-v2',
        [{ name: promptName, arguments: args ?? null }],
        this._operationConfig
      )
    );
  }

  async invalidateToolsCache(): Promise<void> {
    // No-op for stateful servers — tools list is never cached
  }
}
