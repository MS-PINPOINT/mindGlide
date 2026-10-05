/* This worker only downloads fixed public assets before any MRI is selected. */
importScripts(
  "/runtime/pyodide/0.28.3/pyodide.js",
  "/runtime/ort/1.30.0/ort.webgpu.min.js",
  "/tiling.js",
);

const { PATCH, windows, extract, SlabAccumulator } = MindGlideTiling;
let pyodide, session, importance, input, scanInfo, accumulator;
let backend = "CPU",
  busy = false,
  ready = false;
const patchVoxels = PATCH.reduce((a, b) => a * b);
const status = (message, progress) =>
  postMessage({ type: "status", message, progress });

async function verified(entry, message) {
  status(message, null);
  const response = await fetch(`/${entry.path}`, {
    credentials: "omit",
    redirect: "error",
  });
  if (!response.ok)
    throw new Error(
      "A public asset could not be downloaded. Reload and try again.",
    );
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== entry.bytes)
    throw new Error("An engine download was incomplete. Reload and try again.");
  const digest = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
  )
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
  if (digest !== entry.sha256)
    throw new Error("An engine integrity check failed. Reload and try again.");
  return buffer;
}

function closeNetwork() {
  // Defence in depth: all runtime lazy loading has been exercised by warmup.
  // Once ready, this worker has no network API through which to send MRI data.
  const blocked = () => {
    throw new Error(
      "Network access is disabled while processing local MRI data.",
    );
  };
  self.fetch = blocked;
  self.importScripts = blocked;
  if (self.XMLHttpRequest) self.XMLHttpRequest.prototype.open = blocked;
  self.WebSocket = class {
    constructor() {
      blocked();
    }
  };
  self.EventSource = class {
    constructor() {
      blocked();
    }
  };
  self.Worker = class {
    constructor() {
      blocked();
    }
  };
  if (self.WebTransport)
    self.WebTransport = class {
      constructor() {
        blocked();
      }
    };
}

async function warmup() {
  const began = performance.now();
  const tensor = new ort.Tensor("float32", new Float32Array(patchVoxels), [
    1,
    1,
    ...PATCH,
  ]);
  const result = await session.run({ image: tensor });
  if (
    result.logits.dims.join(",") !== [1, 20, ...PATCH].join(",") ||
    !Number.isFinite(result.logits.data[0])
  )
    throw new Error("The local engine did not pass its startup check.");
  result.logits.dispose();
  tensor.dispose();
  return performance.now() - began;
}

async function boot({ preferCPU = false } = {}) {
  if (ready || busy) return;
  busy = true;
  status("Downloading the local engine. Your MRI has not been selected.", null);
  const manifest = await (
    await fetch("/model/manifest.json", {
      credentials: "omit",
      redirect: "error",
    })
  ).json();
  const [modelBuffer, importanceBuffer] = await Promise.all([
    verified(manifest.model, "Downloading the public MindGlide model…"),
    verified(manifest.importance, "Preparing the original patch weighting…"),
  ]);
  importance = new Float32Array(importanceBuffer);
  if (importance.length !== patchVoxels)
    throw new Error("Invalid engine configuration.");
  status("Preparing local image processing…", null);
  pyodide = await loadPyodide({
    indexURL: "/runtime/pyodide/0.28.3/",
    stdout: () => {},
    stderr: () => {},
  });
  await pyodide.loadPackage([
    "numpy",
    "scipy",
    "scikit-image",
    "micropip",
    "packaging",
    "typing-extensions",
  ]);
  const nibabelURL = new URL(
    "/runtime/pyodide/0.28.3/nibabel-5.3.2-py3-none-any.whl",
    self.location.href,
  ).href;
  await pyodide.runPythonAsync(
    `import micropip\nawait micropip.install(${JSON.stringify(nibabelURL)}, deps=False)`,
  );
  const pipeline = await (
    await fetch("/pipeline.py", { credentials: "omit", redirect: "error" })
  ).text();
  await pyodide.runPythonAsync(pipeline);
  ort.env.wasm.wasmPaths = "/runtime/ort/1.30.0/";
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.max(
        1,
        Math.min(8, Math.floor((self.navigator.hardwareConcurrency || 2) / 2)),
      )
    : 1;
  ort.env.logLevel = "error";
  status("Checking the local CPU engine…", null);
  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const cpuSession = session;
  const cpuMillis = await warmup();
  let gpuMillis = null;
  if (!preferCPU && self.navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (
        adapter &&
        adapter.limits.maxStorageBufferBindingSize >= 256 * 1024 * 1024
      ) {
        const device = await adapter.requestDevice({
          requiredLimits: {
            maxStorageBufferBindingSize:
              adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
          },
        });
        ort.env.webgpu.adapter = adapter;
        ort.env.webgpu.device = device;
        status("Checking GPU acceleration on this device…", null);
        session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "basic",
        });
        gpuMillis = await warmup();
        // Browser GPU support alone does not imply faster inference.
        if (gpuMillis < cpuMillis) {
          backend = "GPU";
          await cpuSession.release();
        } else {
          await session.release();
          session = cpuSession;
        }
      }
    } catch (error) {
      postMessage({
        type: "startup-diagnostic",
        message: "GPU startup: " + error.message,
      });
      if (session && session !== cpuSession) {
        try {
          await session.release();
        } catch {}
      }
      session = cpuSession;
      // No scan has been selected at this stage; CPU fallback stays local.
    }
  }
  ready = true;
  busy = false;
  closeNetwork();
  postMessage({
    type: "ready",
    backend,
    threads: backend === "CPU" ? ort.env.wasm.numThreads : null,
    benchmark: { cpuMillis, gpuMillis },
  });
}

async function prepare(buffer) {
  if (!ready || busy) throw new Error("Prepare the local engine first.");
  busy = true;
  if (input) {
    input.fill(0);
    input = null;
  }
  status("Reading and preparing your MRI locally…", null);
  pyodide.globals.set("raw_input", new Uint8Array(buffer));
  try {
    scanInfo = JSON.parse(pyodide.runPython("preprocess(raw_input)"));
    const proxy = pyodide.globals.get("prepared_image");
    const view = proxy.getBuffer("f32");
    input = new Float32Array(view.data);
    view.release();
    proxy.destroy();
    pyodide.runPython(
      "prepared_image.fill(0)\nprepared_image = None\nraw_input = None\ngc.collect()",
    );
  } finally {
    new Uint8Array(buffer).fill(0);
    pyodide.globals.delete("raw_input");
    busy = false;
  }
  postMessage({
    type: "scan-ready",
    info: scanInfo,
    patches: windows(scanInfo.paddedShape).length,
  });
}

async function segment() {
  if (!ready || busy || !input)
    throw new Error("Choose a scan after preparing the engine.");
  busy = true;
  const began = performance.now();
  const shape = scanInfo.paddedShape;
  const list = windows(shape);
  accumulator = new SlabAccumulator(shape, importance);
  const patch = new Float32Array(patchVoxels);
  const tensor = new ort.Tensor("float32", patch, [1, 1, ...PATCH]);
  try {
    for (let i = 0; i < list.length; i++) {
      extract(input, shape, list[i], patch);
      status(
        `Segmenting your brain locally: patch ${i + 1} of ${list.length}`,
        i / list.length,
      );
      const output = await session.run({ image: tensor });
      accumulator.add(output.logits.data, list[i]);
      output.logits.dispose();
      const next = list[i + 1];
      if (!next || next[2] !== list[i][2])
        accumulator.finishBefore(next ? next[2] : shape[2]);
    }
    status("Returning the segmentation to your scan’s original grid…", 0.96);
    pyodide.globals.set("predicted_labels", accumulator.labels);
    const result = JSON.parse(
      pyodide.runPython("recover(predicted_labels.to_py())"),
    );
    const proxy = pyodide.globals.get("segmentation_bytes");
    const view = proxy.getBuffer("u8");
    const segmentation = new Uint8Array(view.data);
    view.release();
    proxy.destroy();
    pyodide.globals.delete("predicted_labels");
    result.seconds = (performance.now() - began) / 1000;
    result.backend = backend;
    postMessage({ type: "result", result, buffer: segmentation.buffer }, [
      segmentation.buffer,
    ]);
  } finally {
    tensor.dispose();
    patch.fill(0);
    if (accumulator) {
      accumulator.clear();
      accumulator = null;
    }
    busy = false;
  }
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "boot") await boot(data);
    else if (data.type === "prepare") await prepare(data.buffer);
    else if (data.type === "segment") await segment();
  } catch (error) {
    busy = false;
    if (!ready)
      postMessage({ type: "startup-diagnostic", message: error.message });
    // No scan data, filenames, headers or stack traces leave the worker.
    const allowed = error?.message?.match(/ValueError: ([^\n]+)/)?.[1];
    postMessage({
      type: "error",
      message:
        allowed ||
        (ready
          ? "Local processing could not finish. Try a smaller scan, close other tabs, or use the local Python tool."
          : "The local engine could not start. Reload, check your connection, or try a current desktop browser."),
    });
  }
};
