/* Fixed original MindGlide patch geometry; no reduced-resolution shortcut. */
(() => {
  const PATCH = [128, 128, 64];
  function starts(size, patch) {
    if (!Number.isSafeInteger(size) || size < patch)
      throw new Error("Invalid working shape");
    const count = Math.ceil((size - patch) / Math.floor(patch / 2)) + 1;
    return Array.from({ length: count }, (_, i) =>
      Math.min(i * Math.floor(patch / 2), size - patch),
    );
  }
  function windows(shape) {
    const axes = shape.map((n, i) => starts(n, PATCH[i]));
    const list = [];
    for (const z of axes[2])
      for (const x of axes[0]) for (const y of axes[1]) list.push([x, y, z]);
    return list;
  }
  function extract(input, shape, start, output) {
    const [x0, y0, z0] = start;
    const [, sy, sz] = shape;
    let target = 0;
    for (let x = 0; x < PATCH[0]; x++)
      for (let y = 0; y < PATCH[1]; y++) {
        const source = ((x + x0) * sy + y + y0) * sz + z0;
        output.set(input.subarray(source, source + PATCH[2]), target);
        target += PATCH[2];
      }
  }
  class SlabAccumulator {
    constructor(shape, importance, classes = 20) {
      this.shape = shape;
      this.classes = classes;
      this.importance = importance;
      this.slabSize = shape[0] * shape[1] * PATCH[2];
      this.scores = new Float32Array(this.slabSize * classes);
      this.counts = new Float32Array(this.slabSize);
      this.labels = new Uint8Array(shape[0] * shape[1] * shape[2]);
      this.finalized = 0;
    }
    add(logits, start) {
      if (logits.length !== this.classes * PATCH[0] * PATCH[1] * PATCH[2])
        throw new Error("Invalid model output");
      const [x0, y0, z0] = start;
      const [, sy] = this.shape;
      const patchVoxels = PATCH[0] * PATCH[1] * PATCH[2];
      for (let x = 0; x < PATCH[0]; x++)
        for (let y = 0; y < PATCH[1]; y++) {
          const base = ((x + x0) * sy + y + y0) * PATCH[2];
          const localBase = (x * PATCH[1] + y) * PATCH[2];
          for (let z = 0; z < PATCH[2]; z++) {
            const index = base + ((z + z0) % PATCH[2]);
            this.counts[index] += this.importance[localBase + z];
          }
          for (let code = 0; code < this.classes; code++) {
            const target = code * this.slabSize + base;
            const source = code * patchVoxels + localBase;
            for (let z = 0; z < PATCH[2]; z++) {
              const index = target + ((z + z0) % PATCH[2]);
              this.scores[index] += Math.fround(
                logits[source + z] * this.importance[localBase + z],
              );
            }
          }
        }
    }
    finishBefore(end) {
      const [sx, sy, sz] = this.shape;
      for (let x = 0; x < sx; x++)
        for (let y = 0; y < sy; y++) {
          const base = (x * sy + y) * PATCH[2];
          for (let z = this.finalized; z < end; z++) {
            const index = base + (z % PATCH[2]);
            const count = this.counts[index];
            if (!(count > 0)) throw new Error("Incomplete prediction coverage");
            let best = -Infinity,
              label = 0;
            for (let code = 0; code < this.classes; code++) {
              const scoreIndex = code * this.slabSize + index;
              const score = Math.fround(this.scores[scoreIndex] / count);
              if (!Number.isFinite(score))
                throw new Error("Non-finite model output");
              if (score > best) {
                best = score;
                label = code;
              }
              this.scores[scoreIndex] = 0;
            }
            this.labels[(x * sy + y) * sz + z] = label;
            this.counts[index] = 0;
          }
        }
      this.finalized = end;
    }
    clear() {
      this.scores.fill(0);
      this.counts.fill(0);
      this.labels.fill(0);
    }
  }
  globalThis.MindGlideTiling = {
    PATCH,
    starts,
    windows,
    extract,
    SlabAccumulator,
  };
})();
