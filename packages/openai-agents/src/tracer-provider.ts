/**
 * Replay-safe OTel TracerProvider factory.
 *
 * Provides {@link createTracerProvider}, the recommended entry point for
 * wiring OpenTelemetry with the Temporal OpenAI Agents plugin. The returned
 * {@link ReplaySafeTracerProvider} wraps a {@link TemporalIdGenerator} that
 * supports pre-seeded trace/span IDs — the mechanism the plugin uses to
 * stitch agent SDK spans into OTel traces with deterministic IDs.
 *
 * Usage:
 * ```ts
 * import { createTracerProvider } from '@temporalio/openai-agents';
 * import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
 * import { trace } from '@opentelemetry/api';
 *
 * const provider = createTracerProvider();
 * provider.addSpanProcessor(new SimpleSpanProcessor(myExporter));
 * trace.setGlobalTracerProvider(provider);
 * ```
 */
import { type IdGenerator, type TracerConfig, BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { TemporalIdGenerator } from './common/tracing-bridge';

/**
 * Options for {@link createTracerProvider}. All fields are optional and
 * forwarded to the underlying `BasicTracerProvider`. The `idGenerator`
 * field, if provided, becomes the fallback generator when no seed is
 * queued (wrapped by {@link TemporalIdGenerator}).
 */
export type CreateTracerProviderOptions = Omit<TracerConfig, 'idGenerator'> & {
  /**
   * Underlying ID generator to delegate to when no seed is queued.
   * Defaults to OTel's `RandomIdGenerator`.
   */
  idGenerator?: IdGenerator;
};

/**
 * An OTel `TracerProvider` that exposes the {@link TemporalIdGenerator}
 * used by the Temporal OpenAI Agents plugin for pre-seeded span IDs.
 *
 * Extends `BasicTracerProvider` so all tracers created from it automatically
 * use the `TemporalIdGenerator` for ID generation. The plugin reads the
 * generator via the public {@link temporalIdGenerator} getter to seed IDs
 * before each `tracer.startSpan()` call — no private-field access needed.
 *
 * Construct via {@link createTracerProvider} rather than directly.
 */
export class ReplaySafeTracerProvider extends BasicTracerProvider {
  private readonly _temporalIdGenerator: TemporalIdGenerator;

  constructor(
    temporalIdGenerator: TemporalIdGenerator,
    config?: Omit<TracerConfig, 'idGenerator'>
  ) {
    super({ ...config, idGenerator: temporalIdGenerator });
    this._temporalIdGenerator = temporalIdGenerator;
  }

  /** The seedable ID generator used by this provider's tracers. */
  get temporalIdGenerator(): TemporalIdGenerator {
    return this._temporalIdGenerator;
  }
}

/**
 * Creates a {@link ReplaySafeTracerProvider} pre-configured with a
 * {@link TemporalIdGenerator}.
 *
 * The returned provider must be registered as the global OTel
 * `TracerProvider` before the Temporal OpenAI Agents plugin initializes:
 *
 * ```ts
 * const provider = createTracerProvider();
 * provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
 * trace.setGlobalTracerProvider(provider);
 * ```
 *
 * Span processors are added post-construction via `addSpanProcessor()`.
 *
 * @param options - Optional configuration for the underlying
 *   `BasicTracerProvider`. The `idGenerator` option, if provided, becomes
 *   the fallback generator when no seed is queued (wraps it with
 *   `TemporalIdGenerator`).
 */
export function createTracerProvider(options?: CreateTracerProviderOptions): ReplaySafeTracerProvider {
  const { idGenerator, ...rest } = options ?? {};
  const generator = new TemporalIdGenerator(idGenerator);
  return new ReplaySafeTracerProvider(generator, rest);
}

