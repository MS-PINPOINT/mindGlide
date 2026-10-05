import { mkdir, readFile, writeFile, copyFile, stat } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const runtime = path.join(root, "public/runtime");
const pyBase = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
const lockPath = path.join(root, "vendor-lock.json");
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
async function download(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(180_000) });
  if (!response.ok)
    throw new Error(`Public asset download failed: HTTP ${response.status}`);
  return Buffer.from(await response.arrayBuffer());
}
async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

await mkdir(runtime, { recursive: true });
if (process.argv.includes("--lock-vendors")) {
  const pyLock = JSON.parse(
    (await download(`${pyBase}pyodide-lock.json`)).toString(),
  );
  const names = new Set();
  function add(name) {
    name = name.toLowerCase().replace(/[_.]+/g, "-");
    if (names.has(name)) return;
    if (!pyLock.packages[name])
      throw new Error(`Unknown pinned package: ${name}`);
    names.add(name);
    pyLock.packages[name].depends.forEach(add);
  }
  [
    "numpy",
    "scipy",
    "scikit-image",
    "micropip",
    "packaging",
    "typing-extensions",
  ].forEach(add);
  let entries = [];
  try {
    entries = (await readJson(lockPath)).entries.filter((entry) =>
      entry.file.startsWith("licenses/"),
    );
  } catch {}
  for (const file of [
    "pyodide.js",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
  ]) {
    const bytes = await download(`${pyBase}${file}`);
    entries.push({
      file: `pyodide/0.28.3/${file}`,
      url: `${pyBase}${file}`,
      bytes: bytes.length,
      sha256: hash(bytes),
    });
    await mkdir(path.join(runtime, "pyodide/0.28.3"), { recursive: true });
    await writeFile(path.join(runtime, "pyodide/0.28.3", file), bytes);
  }
  for (const name of [...names].sort()) {
    const pkg = pyLock.packages[name];
    entries.push({
      file: `pyodide/0.28.3/${pkg.file_name}`,
      url: `${pyBase}${pkg.file_name}`,
      sha256: pkg.sha256,
    });
  }
  const metadata = JSON.parse(
    (await download("https://pypi.org/pypi/nibabel/5.3.2/json")).toString(),
  );
  const wheel = metadata.urls.find((item) =>
    item.filename.endsWith("py3-none-any.whl"),
  );
  if (!wheel) throw new Error("No pure-Python NIfTI reader wheel");
  entries.push({
    file: `pyodide/0.28.3/${wheel.filename}`,
    url: wheel.url,
    sha256: wheel.digests.sha256,
    bytes: wheel.size,
  });
  await writeFile(
    lockPath,
    JSON.stringify({ pyodide: "0.28.3", entries }, null, 2) + "\n",
  );
}
const vendors = await readJson(lockPath);
let total = 0;
for (const entry of vendors.entries) {
  const target = path.join(runtime, entry.file);
  let bytes;
  try {
    bytes = await readFile(target);
  } catch {}
  if (!bytes || hash(bytes) !== entry.sha256) {
    bytes = await download(entry.url);
    if (hash(bytes) !== entry.sha256)
      throw new Error(`Integrity check failed for ${entry.file}`);
    if (entry.bytes && bytes.length !== entry.bytes)
      throw new Error("Public asset size mismatch");
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, bytes);
  }
  total += bytes.length;
}
const ortPath = path.join(root, "node_modules/onnxruntime-web/dist");
const ortTarget = path.join(runtime, "ort/1.30.0");
await mkdir(ortTarget, { recursive: true });
for (const name of [
  "ort.webgpu.min.js",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.jspi.mjs",
  "ort-wasm-simd-threaded.jspi.wasm",
]) {
  await copyFile(path.join(ortPath, name), path.join(ortTarget, name));
}
const modelLockPath = path.join(root, "model-lock.json");
let model;
try {
  model = await readJson(modelLockPath);
} catch {
  console.log(
    `Scientific runtime prepared (${(total / 1048576).toFixed(1)} MiB). Model lock awaits validated export.`,
  );
  process.exit(0);
}
const publicManifest = { ...model };
for (const key of ["model", "importance", "example"]) {
  const entry = model[key];
  const target = path.join(root, "public", entry.path);
  let bytes;
  try {
    bytes = await readFile(target);
  } catch {}
  if (!bytes || hash(bytes) !== entry.sha256) {
    bytes = await download(entry.url);
    if (hash(bytes) !== entry.sha256 || bytes.length !== entry.bytes)
      throw new Error(`Integrity check failed: ${key}`);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, bytes);
  }
  delete publicManifest[key].url;
}
await writeFile(
  path.join(root, "public/model/manifest.json"),
  JSON.stringify(publicManifest, null, 2) + "\n",
);
console.log(
  "Pinned runtime, model and public example prepared; no credentials used.",
);
await import("./notices.mjs");
