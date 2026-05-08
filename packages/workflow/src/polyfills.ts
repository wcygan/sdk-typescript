/**
 * Web-API polyfills for the workflow V8 sandbox.
 *
 * Installs: Headers, ReadableStream (+ related stream APIs), structuredClone,
 * and crypto.randomUUID. The randomUUID implementation is wired through
 * Temporal's deterministic PRNG (uuid4()) for replay safety.
 *
 * Import this module for side-effects:
 *
 *   import '@temporalio/workflow/polyfills';
 */

import { Headers } from 'headers-polyfill';
import { inWorkflowContext, uuid4 } from './workflow';

if (inWorkflowContext()) {
  // Headers polyfill
  if (typeof globalThis.Headers === 'undefined') {
    (globalThis as any).Headers = Headers;
  }

  // ReadableStream polyfill
  // eslint-disable-next-line @typescript-eslint/no-require-imports,import/no-unassigned-import
  require('web-streams-polyfill/polyfill');

  // structuredClone polyfill
  if (!('structuredClone' in globalThis)) {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const sc = require('@ungap/structured-clone');
    (globalThis as any).structuredClone = sc.default;
  }

  // crypto.randomUUID — uses Temporal's deterministic uuid4() which is backed
  // by a per-workflow seeded PRNG, ensuring replay safety and per-workflow isolation.
  if (typeof (globalThis as any).crypto === 'undefined') {
    (globalThis as any).crypto = {};
  }
  if (!(globalThis as any).crypto.randomUUID) {
    (globalThis as any).crypto.randomUUID = (): string => uuid4();
  }
}
