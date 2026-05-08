import { ApplicationFailure } from '@temporalio/common';
import { Context, heartbeat, activityInfo } from '@temporalio/activity';
import type { NativeConnection } from '@temporalio/worker';
import type { MCPPromptDefinition } from '../workflow/mcp-client';
import type { MCPToolDefinition, MCPCallToolResult } from './mcp-provider';

/**
 * An MCP server interface matching the subset used by stateful providers.
 * This is the shape that `serverFactory` must return.
 */
export interface StatefulMCPServer {
  connect(): Promise<void>;
  cleanup(): Promise<void>;
  listTools(): Promise<MCPToolDefinition[]>;
  callTool(toolName: string, args: Record<string, unknown> | null): Promise<MCPCallToolResult[]>;
  listPrompts?(): Promise<MCPPromptDefinition[]>;
  getPrompt?(name: string, args: Record<string, unknown> | null): Promise<unknown>;
}

interface ServerSessionArgs {
  factoryArgument?: unknown;
}

interface CallToolArgs {
  toolName: string;
  args: Record<string, unknown> | null;
}

interface GetPromptArgs {
  name: string;
  arguments: Record<string, unknown> | null;
}

/**
 * A stateful MCP server provider for Temporal workflows.
 *
 * Maintains a persistent MCP server connection per workflow run via a
 * dedicated in-process worker on a per-run task queue. The server is
 * created from the provided factory, connected, and kept alive for the
 * duration of the workflow's use. Operations (listTools, callTool, etc.)
 * execute on the dedicated worker, preserving server-side state.
 *
 * Users must handle `ApplicationFailure` with type `"DedicatedWorkerFailure"`
 * when the dedicated worker fails to start or misses a heartbeat.
 */
export class StatefulMCPServerProvider {
  private readonly _name: string;
  private readonly _servers: Map<string, StatefulMCPServer> = new Map();
  private readonly _serverFactory: (factoryArgument: unknown | undefined) => StatefulMCPServer;
  private readonly _nativeConnection: NativeConnection;

  /**
   * @param name - Server name. The internal name is `${name}-stateful` to avoid
   *   collisions with stateless providers using the same base name.
   * @param serverFactory - Factory that creates MCP server instances. Called once per
   *   workflow run. Must return a fresh instance each time.
   * @param nativeConnection - NativeConnection for the dedicated per-run worker.
   *   The dedicated worker uses this connection to communicate with the Temporal cluster.
   *   Typically, pass the same NativeConnection used by the main worker.
   */
  constructor(
    name: string,
    serverFactory: (factoryArgument: unknown | undefined) => StatefulMCPServer,
    nativeConnection: NativeConnection
  ) {
    this._name = `${name}-stateful`;
    this._serverFactory = serverFactory;
    this._nativeConnection = nativeConnection;
  }

  get name(): string {
    return this._name;
  }

  _getActivities(): Record<string, (...args: any[]) => Promise<unknown>> {
    const serverId = (): string => {
      const info = activityInfo();
      return `${this._name}@${info.workflowExecution.runId}`;
    };

    // Dedicated activities that run on the per-run task queue
    const listTools = async (): Promise<MCPToolDefinition[]> => {
      const server = this._servers.get(serverId());
      if (!server) {
        throw ApplicationFailure.create({
          message: `No active server for ${serverId()}`,
          type: 'StatefulMCPServerNotFound',
          nonRetryable: true,
        });
      }
      return server.listTools();
    };

    const callTool = async (input: CallToolArgs): Promise<MCPCallToolResult[]> => {
      const server = this._servers.get(serverId());
      if (!server) {
        throw ApplicationFailure.create({
          message: `No active server for ${serverId()}`,
          type: 'StatefulMCPServerNotFound',
          nonRetryable: true,
        });
      }
      return server.callTool(input.toolName, input.args);
    };

    const callToolDeprecated = async (
      toolName: string,
      args: Record<string, unknown> | null
    ): Promise<MCPCallToolResult[]> => {
      return callTool({ toolName, args });
    };

    const listPrompts = async (): Promise<MCPPromptDefinition[]> => {
      const server = this._servers.get(serverId());
      if (!server) {
        throw ApplicationFailure.create({
          message: `No active server for ${serverId()}`,
          type: 'StatefulMCPServerNotFound',
          nonRetryable: true,
        });
      }
      return server.listPrompts?.() ?? [];
    };

    const getPrompt = async (input: GetPromptArgs): Promise<unknown> => {
      const server = this._servers.get(serverId());
      if (!server) {
        throw ApplicationFailure.create({
          message: `No active server for ${serverId()}`,
          type: 'StatefulMCPServerNotFound',
          nonRetryable: true,
        });
      }
      return server.getPrompt?.(input.name, input.arguments) ?? null;
    };

    const getPromptDeprecated = async (
      name: string,
      args: Record<string, unknown> | null
    ): Promise<unknown> => {
      return getPrompt({ name, arguments: args });
    };

    // The long-running session activity registered on the MAIN worker's task queue.
    // It creates a server, connects it, spins up a dedicated worker with the
    // operation activities, and awaits until cancelled.
    const serverSession = async (input?: ServerSessionArgs): Promise<void> => {
      const info = activityInfo();
      const sid = `${this._name}@${info.workflowExecution.runId}`;

      // Heartbeat immediately so slow server.connect() doesn't cause a heartbeat timeout
      const heartbeatInterval = setInterval(() => {
        heartbeat();
      }, 30_000);

      try {
        if (this._servers.has(sid)) {
          throw ApplicationFailure.create({
            message:
              'Cannot connect to an already running server. Use a distinct name if running multiple servers in one workflow.',
            type: 'StatefulMCPServerAlreadyRunning',
            nonRetryable: true,
          });
        }

        const server = this._serverFactory(input?.factoryArgument);

        try {
          this._servers.set(sid, server);
          await server.connect();

          const { Worker } = await import('@temporalio/worker');

          const dedicatedTaskQueue = sid;

          const dedicatedActivities: Record<string, (...args: any[]) => Promise<unknown>> = {
            [`${this._name}-list-tools`]: listTools,
            [`${this._name}-call-tool-v2`]: callTool,
            [`${this._name}-call-tool`]: callToolDeprecated,
            [`${this._name}-list-prompts`]: listPrompts,
            [`${this._name}-get-prompt-v2`]: getPrompt,
            [`${this._name}-get-prompt`]: getPromptDeprecated,
          };

          const dedicatedWorker = await Worker.create({
            connection: this._nativeConnection,
            taskQueue: dedicatedTaskQueue,
            activities: dedicatedActivities,
            maxConcurrentActivityTaskExecutions: 1,
          });

          // Shut down the dedicated worker when the session activity is cancelled
          const ctx = Context.current();
          ctx.cancelled.catch(() => {
            dedicatedWorker.shutdown();
          });

          await dedicatedWorker.run();
        } finally {
          await server.cleanup();
          this._servers.delete(sid);
        }
      } finally {
        clearInterval(heartbeatInterval);
      }
    };

    // Only the session activity is registered on the main worker.
    // The dedicated operation activities are registered on the per-run worker
    // inside the session activity body.
    return {
      [`${this._name}-server-session`]: serverSession,
    };
  }
}
