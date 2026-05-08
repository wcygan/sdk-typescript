import { TemporalFailure } from '@temporalio/common';

/**
 * Dig through nested error wrappers to find the first Temporal-specific failure.
 *
 * Errors thrown out of an agent run are typically wrapped several layers deep —
 * guardrail wrappers, runner wrappers, `ApplicationFailure.create({ cause })`, etc.
 * To branch on Temporal failure types (`ActivityFailure`, `CancelledFailure`, ...),
 * callers need to peel back those layers.
 *
 * This helper does that walk: it descends `error.cause` chains and (for
 * `AggregateError`) `error.errors` arrays, tracking visited nodes with a `Set`
 * for cycle safety. Returns the first `TemporalFailure` it finds, or `undefined`
 * if none exists in the graph.
 */
export function unwrapTemporalFailure(error: unknown): TemporalFailure | undefined {
  const visited = new Set<unknown>();
  const stack: unknown[] = [error];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current || typeof current !== 'object' || visited.has(current)) continue;
    visited.add(current);
    if (current instanceof TemporalFailure) return current;
    if (current instanceof AggregateError) {
      for (const inner of current.errors) {
        stack.push(inner);
      }
    }
    stack.push((current as any).cause);
  }
  return undefined;
}
