import { enhanceSurface } from './realism-surface.js';

self.onmessage = ({ data }) => {
  try {
    const result = enhanceSurface(data, (message) =>
      self.postMessage({ progress: message }),
    );
    self.postMessage(result, [result.pixels.buffer]);
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
