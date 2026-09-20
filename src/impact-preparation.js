// One worker per loaded head. Jobs contain independent, non-damaging live
// contacts; persistent clay/damage continues through the ordered local path.
import { impactCacheKey } from './impact-cache.js';

export class ImpactPreparation {
  constructor(
    model,
    {
      makeWorker = () =>
        new Worker(new URL('./impact-worker.js', import.meta.url), { type: 'module' }),
    } = {},
  ) {
    this.worker = makeWorker();
    this.pending = new Map();
    this.cache = new Map();
    this.sequence = 0;
    this.epoch = 0;
    this.disposed = false;
    this.worker.onmessage = ({ data }) => {
      if (this.disposed || data.epoch !== this.epoch) return;
      if (data.type === 'cache') {
        this.remember(data.key, data);
        return;
      }
      if (data.type === 'ready') {
        if (data.error) this.fail();
        else this.resolveReady?.();
        return;
      }
      const job = this.pending.get(data.id);
      if (!job) return;
      if (data.stage === 'complete' && data.event) this.remember(job.key, data);
      if (data.stage !== 'preview') this.pending.delete(data.id);
      job.callback(data);
    };
    this.worker.onerror = () => this.fail();
    this.configure(model);
  }

  configure(model) {
    this.invalidate();
    this.cache.clear();
    this.ready = new Promise((resolve) => {
      this.resolveReady = resolve;
    });
    this.worker.postMessage({ type: 'init', epoch: this.epoch, ...model });
  }

  request(input, softness, reactionEnabled, callback) {
    if (this.disposed || this.failed) return false;
    const key = impactCacheKey(input, softness, reactionEnabled);
    const cached = this.cache.get(key);
    if (cached) {
      callback({
        ...cached,
        stage: 'complete',
        milliseconds: 0,
        cached: true,
        event: { ...cached.event, age: 0, committed: 0 },
      });
      return true;
    }
    if (this.pending.size >= 4) return false;
    const id = ++this.sequence;
    this.pending.set(id, { callback, key });
    this.worker.postMessage({
      type: 'impact',
      epoch: this.epoch,
      id,
      input,
      softness,
      reactionEnabled,
    });
    return true;
  }

  invalidate() {
    this.resolveReady?.();
    this.epoch++;
    this.pending.clear();
  }

  remember(key, result) {
    this.cache.delete(key);
    this.cache.set(key, result);
    if (this.cache.size > 24) this.cache.delete(this.cache.keys().next().value);
  }

  fail() {
    if (this.disposed || this.failed) return;
    this.failed = true;
    this.resolveReady?.();
    this.worker.terminate();
    const callbacks = [...this.pending.values()];
    this.pending.clear();
    for (const { callback } of callbacks)
      callback({ error: 'Impact worker unavailable' });
  }

  dispose() {
    this.disposed = true;
    this.invalidate();
    this.worker.terminate();
  }
}
