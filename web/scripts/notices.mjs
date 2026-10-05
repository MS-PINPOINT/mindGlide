import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
const root = path.resolve(import.meta.dirname, "..");
const chunks = [
  "Third-party notices for MindGlide browser edition\n\nRuntime wheel archives include their own distribution notices. Original model terms are unchanged. Public MNI attribution is in attribution.html.\n",
];
async function visit(directory) {
  for (const item of await readdir(directory, { withFileTypes: true })) {
    if (!item.isDirectory() || item.name.startsWith(".")) continue;
    const full = path.join(directory, item.name);
    if (item.name.startsWith("@")) {
      await visit(full);
      continue;
    }
    let pkg;
    try {
      pkg = JSON.parse(await readFile(path.join(full, "package.json")));
    } catch {
      continue;
    }
    chunks.push(
      `\n=== ${pkg.name} ${pkg.version} (${typeof pkg.license === "string" ? pkg.license : "see upstream"}) ===\n`,
    );
    for (const file of await readdir(full))
      if (/^(licen[sc]e|copying|notice)(\.|$)/i.test(file)) {
        try {
          chunks.push(await readFile(path.join(full, file), "utf8"));
        } catch {}
      }
  }
}
await visit(path.join(root, "node_modules"));
for (const file of [
  "onnxruntime-LICENSE",
  "onnxruntime-ThirdPartyNotices.txt",
  "niivue-LICENSE",
  "pyodide-LICENSE",
  "cpython-LICENSE",
])
  chunks.push(
    `\n=== ${file} ===\n${await readFile(path.join(root, "public/runtime/licenses", file), "utf8")}`,
  );
await writeFile(
  path.join(root, "public/THIRD-PARTY-NOTICES.txt"),
  chunks.join("\n"),
);
