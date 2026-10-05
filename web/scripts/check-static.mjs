import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
const root = path.resolve(import.meta.dirname, "..");
const dist = path.join(root, "dist");
async function files(directory) {
  const list = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) list.push(...(await files(full)));
    else list.push(full);
  }
  return list;
}
const output = await files(dist);
for (const file of output) {
  const relative = path.relative(dist, file);
  if (
    /(^|\/)(api|\.vercel|\.git|\.env|node_modules|validation-data)(\/|$)|\.(pem|key|pt|npz|map)$/.test(
      relative,
    )
  )
    throw new Error(`Unexpected deployment file: ${relative}`);
}
for (const name of [
  "index.html",
  "privacy.html",
  "attribution.html",
  "worker.js",
  "pipeline.py",
  "tiling.js",
  "model/manifest.json",
])
  await stat(path.join(dist, name));
const manifest = JSON.parse(
  await readFile(path.join(dist, "model/manifest.json")),
);
for (const key of ["model", "importance", "example"]) {
  const entry = manifest[key];
  const size = (await stat(path.join(dist, entry.path))).size;
  if (size !== entry.bytes) throw new Error("Incomplete static artifact");
}
const config = JSON.parse(await readFile(path.join(root, "vercel.json")));
if (
  config.framework !== null ||
  config.outputDirectory !== "dist" ||
  config.functions ||
  config.rewrites
)
  throw new Error(
    "This project must remain static, with no functions or proxies",
  );
const bytes = (await Promise.all(output.map((file) => stat(file)))).reduce(
  (n, info) => n + info.size,
  0,
);
console.log(
  `Static deployment verified: ${output.length} public files, ${(bytes / 1048576).toFixed(1)} MiB, no Function routes.`,
);
