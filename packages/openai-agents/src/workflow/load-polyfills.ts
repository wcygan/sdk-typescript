// Web-API polyfills (Headers, ReadableStream, structuredClone, crypto.randomUUID, EventTarget, Event, CustomEvent)
// eslint-disable-next-line import/no-unassigned-import
import '@temporalio/workflow/polyfills';

import { inWorkflowContext } from '@temporalio/workflow';

if (inWorkflowContext()) {
  // EventTarget polyfill — agents-core uses EventTarget internally for event handling
  if (typeof (globalThis as any).EventTarget === 'undefined') {
    (globalThis as any).EventTarget = class EventTargetPolyfill {
      _listeners: Record<string, Array<(event: any) => void>> = {};

      addEventListener(type: string, listener: (event: any) => void): void {
        if (!this._listeners[type]) this._listeners[type] = [];
        this._listeners[type].push(listener);
      }

      removeEventListener(type: string, listener: (event: any) => void): void {
        const arr = this._listeners[type];
        if (arr) this._listeners[type] = arr.filter((l) => l !== listener);
      }

      dispatchEvent(event: any): boolean {
        (event as any).target = this;
        (event as any).currentTarget = this;
        const arr = this._listeners[event.type];
        if (arr) {
          arr.forEach((l) => {
            try {
              l(event);
            } catch {
              // Isolate listener errors — one bad listener shouldn't break dispatch
            }
          });
        }
        return true;
      }
    };
  }

  // Event polyfill
  if (typeof (globalThis as any).Event === 'undefined') {
    (globalThis as any).Event = class EventPolyfill {
      type: string;
      bubbles: boolean;
      cancelable: boolean;

      constructor(type: string, opts?: { bubbles?: boolean; cancelable?: boolean }) {
        this.type = type;
        this.bubbles = opts?.bubbles ?? false;
        this.cancelable = opts?.cancelable ?? false;
      }
    };
  }

  // CustomEvent polyfill
  if (typeof (globalThis as any).CustomEvent === 'undefined') {
    const EventClass = (globalThis as any).Event;
    (globalThis as any).CustomEvent = class CustomEventPolyfill extends EventClass {
      detail: any;

      constructor(type: string, opts?: { bubbles?: boolean; cancelable?: boolean; detail?: any }) {
        super(type, opts);
        this.detail = opts?.detail ?? null;
      }
    };
  }
}
