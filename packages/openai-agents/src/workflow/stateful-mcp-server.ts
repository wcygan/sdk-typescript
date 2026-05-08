import { StatefulMCPServerReference, type StatefulMcpServerOptions } from './stateful-mcp-client';

/**
 * Creates a stateful MCP server handle for use in Temporal workflows.
 *
 * Unlike `statelessMcpServer`, this maintains a persistent MCP connection
 * throughout the workflow execution via a dedicated in-process worker on a
 * per-run task queue. All tool and prompt operations are routed through
 * that persistent connection, preserving server-side state across calls.
 *
 * This is useful when the MCP server needs to maintain state between
 * operations (e.g., a database session, authenticated connection, or
 * accumulated context). The tradeoff is additional resource overhead and
 * the need to handle dedicated-worker failures.
 *
 * **Error handling**: If the dedicated worker fails (startup timeout or
 * missed heartbeat), operations will throw `ApplicationFailure` with
 * type `"DedicatedWorkerFailure"`. Callers must handle this and decide
 * whether to restart or abort.
 *
 * @param name - Server name. Must match the name used in the
 *   `StatefulMCPServerProvider` registered on the worker side.
 * @param options - Optional configuration for activity timeouts and factory arguments.
 * @returns A `TemporalMCPServer` with `connect()` and `cleanup()` lifecycle methods.
 *
 * @example
 * ```ts
 * const server = statefulMcpServer('my-db-server');
 * await server.connect();
 * try {
 *   const agent = new Agent({ mcpServers: [server], ... });
 *   const result = await runner.run(agent, prompt);
 * } finally {
 *   await server.cleanup();
 * }
 * ```
 */
export function statefulMcpServer(
  name: string,
  options?: StatefulMcpServerOptions
): StatefulMCPServerReference {
  return new StatefulMCPServerReference(name, options);
}
