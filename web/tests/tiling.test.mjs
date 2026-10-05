import test from "node:test";
import assert from "node:assert/strict";
import "../public/tiling.js";
const { PATCH, starts, windows, extract, SlabAccumulator } =
  globalThis.MindGlideTiling;
const patchSize = PATCH.reduce((a, b) => a * b);
test("half-overlap windows include irregular final edges exactly once", () => {
  assert.deepEqual(starts(193, 128), [0, 64, 65]);
  assert.deepEqual(starts(97, 64), [0, 32, 33]);
  assert.deepEqual(starts(128, 128), [0]);
  assert.throws(() => starts(127, 128));
  for (const size of [64, 65, 96, 97, 192, 229]) {
    const coverage = new Uint8Array(size);
    for (const start of starts(size, 64))
      for (let z = start; z < start + 64; z++) coverage[z]++;
    assert.ok(coverage.every((n) => n > 0));
  }
});
test("patch extraction preserves C-order voxel coordinates", () => {
  const shape = [129, 130, 65];
  const input = Float32Array.from(
    { length: shape.reduce((a, b) => a * b) },
    (_, i) => i,
  );
  const out = new Float32Array(patchSize);
  extract(input, shape, [1, 2, 1], out);
  assert.equal(out[0], (1 * 130 + 2) * 65 + 1);
  assert.equal(out.at(-1), (128 * 130 + 129) * 65 + 64);
});
test("bounded ring aggregation matches a full weighted volume through reuse and ties", () => {
  const shape = [129, 130, 129],
    classes = 3;
  const voxels = shape.reduce((a, b) => a * b);
  const importance = Float32Array.from(
    { length: patchSize },
    (_, i) => 0.01 + (i % 17) / 17,
  );
  const slab = new SlabAccumulator(shape, importance, classes);
  assert.equal(slab.scores.length, classes * shape[0] * shape[1] * 64);
  const scores = new Float32Array(classes * voxels),
    counts = new Float32Array(voxels);
  const list = windows(shape);
  for (let w = 0; w < list.length; w++) {
    const [x0, y0, z0] = list[w];
    const logits = new Float32Array(classes * patchSize);
    for (let x = 0; x < 128; x++)
      for (let y = 0; y < 128; y++)
        for (let z = 0; z < 64; z++) {
          const local = (x * 128 + y) * 64 + z,
            global = ((x + x0) * shape[1] + y + y0) * shape[2] + z + z0;
          counts[global] += importance[local];
          for (let c = 0; c < classes; c++) {
            // Deliberate exact ties, window-dependent scores and ring reuse.
            const score =
              global % 7 === 0 ? 1 : Math.sin(global * 0.01 + c * 3 + w * 0.7);
            logits[c * patchSize + local] = score;
            scores[c * voxels + global] += Math.fround(
              logits[c * patchSize + local] * importance[local],
            );
          }
        }
    slab.add(logits, list[w]);
    const next = list[w + 1];
    if (!next || next[2] !== z0) slab.finishBefore(next ? next[2] : shape[2]);
  }
  for (let i = 0; i < voxels; i++) {
    let label = 0,
      best = -Infinity;
    for (let c = 0; c < classes; c++) {
      const score = Math.fround(scores[c * voxels + i] / counts[i]);
      if (score > best) {
        best = score;
        label = c;
      }
    }
    assert.equal(slab.labels[i], label);
  }
  slab.clear();
  assert.ok(slab.labels.every((n) => n === 0));
});
test("incomplete and non-finite outputs fail instead of producing a result", () => {
  const slab = new SlabAccumulator(
    PATCH,
    new Float32Array(patchSize).fill(1),
    2,
  );
  assert.throws(() => slab.finishBefore(1), /Incomplete/);
  assert.throws(
    () => slab.add(new Float32Array(10), [0, 0, 0]),
    /Invalid model/,
  );
  const bad = new Float32Array(2 * patchSize);
  bad[0] = NaN;
  slab.add(bad, [0, 0, 0]);
  assert.throws(() => slab.finishBefore(64), /Non-finite/);
});
