import { Niivue, NVImage } from "@niivue/niivue";
import "./style.css";

const $ = (id) => document.getElementById(id);
const labels = [
  "Background",
  "CSF",
  "Third, fourth & fifth ventricles",
  "Deep grey matter",
  "Pons",
  "Brainstem",
  "Cerebellum",
  "Temporal lobe",
  "Temporal horn of lateral ventricle",
  "Lateral ventricle",
  "Optic chiasm",
  "Cerebellar vermis",
  "Corpus callosum",
  "White matter",
  "Frontal lobe grey matter",
  "Limbic cortex grey matter",
  "Parietal lobe grey matter",
  "Occipital lobe grey matter",
  "Lesion",
  "Ventral diencephalon",
];
const colors = [
  "000000",
  "77bde8",
  "b5dbea",
  "f4c46e",
  "f2a777",
  "e088ba",
  "a398ea",
  "6baedf",
  "78d3cb",
  "75bed4",
  "ded978",
  "b5a6eb",
  "b9c7ee",
  "ece5bd",
  "8aceb0",
  "e5b57e",
  "79becc",
  "c99fd7",
  "ff6476",
  "d5cb78",
];
let worker,
  engineReady = false,
  busy = false,
  hasScan = false;
let nv, original, segmentation, result, publicExample;
let generation = 0;
let exportURLs = [];
const MAX_FILE = 128 * 1024 * 1024;

function status(message, progress = undefined, error = false) {
  $("status").textContent = message;
  $("status").parentElement.classList.toggle("error", error);
  $("progress").hidden = progress === undefined;
  if (progress === null) $("progress").removeAttribute("value");
  else if (progress !== undefined) $("progress").value = progress;
}
function controls() {
  $("boot").disabled = busy || engineReady;
  $("engine-mode").disabled = busy || engineReady;
  $("choose").disabled = !engineReady || busy;
  $("file").disabled = !engineReady || busy;
  $("example").disabled = !engineReady || busy || !publicExample;
  $("run").disabled = !hasScan || busy || !$("research").checked;
  $("clear").disabled = !worker && !hasScan;
  $("opacity").disabled = !segmentation || busy;
}
function sealNetwork() {
  // Applied only after every public runtime asset and the example are ready.
  const policy = document.createElement("meta");
  policy.httpEquiv = "Content-Security-Policy";
  policy.content =
    "default-src 'none'; script-src 'none'; worker-src 'none'; connect-src 'none'; img-src data: blob:; font-src data:; style-src 'unsafe-inline'; form-action 'none'";
  document.head.append(policy);
  const blocked = () => {
    throw new Error("Network access is disabled for this local session.");
  };
  window.fetch = blocked;
  XMLHttpRequest.prototype.open = blocked;
  window.WebSocket = class {
    constructor() {
      blocked();
    }
  };
  window.EventSource = class {
    constructor() {
      blocked();
    }
  };
  window.Worker = class {
    constructor() {
      blocked();
    }
  };
  if (window.WebTransport)
    window.WebTransport = class {
      constructor() {
        blocked();
      }
    };
}
async function boot() {
  busy = true;
  controls();
  const thisGeneration = ++generation;
  $("engine-badge").textContent = "Preparing…";
  status("Downloading public engine assets. No MRI is selected yet.", null);
  try {
    const manifest = await (
      await fetch("/model/manifest.json", {
        credentials: "omit",
        redirect: "error",
      })
    ).json();
    $("download-size").textContent =
      `Public model: ${Math.ceil(manifest.model.bytes / 1048576)} MB, plus image-processing libraries. The browser may cache these public files.`;
    const response = await fetch(`/${manifest.example.path}`, {
      credentials: "omit",
      redirect: "error",
    });
    if (!response.ok) throw new Error("Example download failed");
    publicExample = await response.arrayBuffer();
    const digest = [
      ...new Uint8Array(await crypto.subtle.digest("SHA-256", publicExample)),
    ]
      .map((n) => n.toString(16).padStart(2, "0"))
      .join("");
    if (
      publicExample.byteLength !== manifest.example.bytes ||
      digest !== manifest.example.sha256
    )
      throw new Error("Example integrity check failed");
    if (thisGeneration !== generation) return;
    worker = new Worker("/worker.js");
    worker.onmessage = async ({ data }) => {
      if (thisGeneration !== generation) return;
      if (data.type === "startup-diagnostic")
        console.error("Public runtime startup failed:", data.message);
      if (data.type === "status") status(data.message, data.progress);
      if (data.type === "ready") {
        engineReady = true;
        busy = false;
        sealNetwork();
        $("engine-badge").textContent = "Local engine ready";
        $("backend").textContent =
          data.backend === "GPU"
            ? "GPU accelerated · FP32"
            : `Local CPU · ${data.threads} thread${data.threads === 1 ? "" : "s"} · FP32`;
        $("step-engine").classList.remove("active");
        $("step-engine").classList.add("complete");
        $("step-scan").classList.add("active");
        status(
          "Engine ready. Network access is now disabled for processing. Choose a local MRI or the public example.",
        );
      }
      if (data.type === "scan-ready") {
        try {
          await showOriginal();
          hasScan = true;
          $("scan-details").hidden = false;
          $("scan-details").textContent =
            `${data.info.shape.join(" × ")} voxels · ${data.info.spacing.map((x) => x.toFixed(2)).join(" × ")} mm · ${data.patches} patches${data.info.unitsAssumed ? " · units absent: assumes mm" : ""}`;
          $("step-scan").classList.remove("active");
          $("step-scan").classList.add("complete");
          $("step-run").classList.add("active");
          $("view-state").textContent = "Original MRI · local";
          status(
            "Your MRI is ready. Review the research-use note, then start local segmentation.",
          );
        } catch {
          status(
            "The scan was prepared, but this browser could not display it. Try a current desktop browser.",
            undefined,
            true,
          );
        }
        busy = false;
      }
      if (data.type === "result") {
        result = data.result;
        segmentation = new Uint8Array(data.buffer);
        prepareExports();
        try {
          const overlay = await NVImage.loadFromFile({
            file: new File([segmentation], "segmentation.nii"),
            opacity: 0.45,
          });
          const rgb = colors.map((hex) =>
            [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)),
          );
          overlay.setColormapLabel({
            I: labels.map((_, i) => i),
            R: rgb.map((v) => v[0]),
            G: rgb.map((v) => v[1]),
            B: rgb.map((v) => v[2]),
            A: labels.map((_, i) => (i ? 255 : 0)),
            labels,
          });
          nv.addVolume(overlay);
          $("exports").hidden = false;
          $("view-state").textContent = "MRI + segmentation · local";
          $("result-summary").hidden = false;
          const time =
            result.seconds >= 60
              ? `${(result.seconds / 60).toFixed(1)} min`
              : `${Math.round(result.seconds)} sec`;
          $("result-summary").textContent = result.empty
            ? "No foreground was found. Check the input and inspect the result; this is not a medical finding."
            : `Finished in ${time} on your ${result.backend}. Overlay and exports use your MRI’s original grid.`;
          $("volume-rows").replaceChildren();
          labels.slice(1).forEach((name, index) => {
            const code = index + 1;
            const row = document.createElement("tr");
            for (const value of [
              String(code),
              name,
              (result.volumes[code] / 1000).toFixed(3),
            ]) {
              const cell = document.createElement("td");
              cell.textContent = value;
              row.append(cell);
            }
            row.children[1].style.setProperty(
              "--region-color",
              `#${colors[code]}`,
            );
            $("volume-rows").append(row);
          });
          $("regions").hidden = false;
          $("step-run").classList.remove("active");
          $("step-run").classList.add("complete");
          status(
            "Segmentation complete. Explore the overlay or download your local results.",
          );
        } catch {
          status(
            "Segmentation finished, but the overlay could not be displayed. You can download the local NIfTI result.",
            undefined,
            true,
          );
          $("exports").hidden = false;
        }
        busy = false;
        hasScan = false;
      }
      if (data.type === "error") {
        busy = false;
        status(data.message, undefined, true);
        if (!engineReady) $("engine-badge").textContent = "Engine unavailable";
      }
      controls();
    };
    worker.onerror = () => {
      busy = false;
      status(
        "The local worker stopped. This device may have run out of memory. Clear and reset, or use the local Python tool.",
        undefined,
        true,
      );
      controls();
    };
    worker.postMessage({
      type: "boot",
      preferCPU: $("engine-mode").value === "cpu",
    });
  } catch (error) {
    if (thisGeneration !== generation) return;
    console.error("Public engine asset preparation failed:", error.message);
    busy = false;
    $("engine-badge").textContent = "Engine unavailable";
    status(
      "Public engine assets could not be downloaded. Check your connection and reload.",
      undefined,
      true,
    );
    controls();
  }
}
function clearVolumes() {
  if (!nv) return;
  for (const volume of [...nv.volumes]) {
    if (volume.img?.fill) volume.img.fill(0);
    nv.removeVolume(volume);
  }
}
async function showOriginal() {
  clearVolumes();
  $("placeholder").hidden = true;
  $("brain").hidden = false;
  if (!nv) {
    nv = new Niivue({
      backColor: [0.025, 0.04, 0.065, 1],
      crosshairColor: [0.5, 0.85, 0.8, 0.8],
      isColorbar: false,
      dragAndDropEnabled: false,
      logLevel: "silent",
      loadingText: "",
      multiplanarShowRender: 0,
    });
    await nv.attachToCanvas($("brain"));
    nv.setSliceType(nv.sliceTypeMultiplanar);
  }
  // FileReader only: no blob URL fetch, filename propagation or remote loader.
  const volume = await NVImage.loadFromFile({
    file: new File([original], "local-mri.nii"),
    name: "Local MRI",
    colormap: "gray",
  });
  nv.addVolume(volume);
}
async function choose(file) {
  if (!engineReady || busy || !file) return;
  if (
    !/\.nii(?:\.gz)?$/i.test(file.name) ||
    file.size < 2 ||
    file.size > MAX_FILE
  ) {
    status(
      "Choose a .nii or .nii.gz file under 128 MB containing one 3D brain MRI.",
      undefined,
      true,
    );
    return;
  }
  busy = true;
  hasScan = false;
  controls();
  clearVolumes();
  if (original) new Uint8Array(original).fill(0);
  if (segmentation) segmentation.fill(0);
  segmentation = result = null;
  clearExports();
  $("exports").hidden = true;
  $("regions").hidden = true;
  $("result-summary").hidden = true;
  try {
    original = await file.arrayBuffer();
    const forWorker = original.slice(0);
    worker.postMessage({ type: "prepare", buffer: forWorker }, [forWorker]);
    status("Preparing this scan locally…", null);
  } catch {
    busy = false;
    status("The local file could not be read.", undefined, true);
    controls();
  }
}
function exportLink(id, bytes, name, type) {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  exportURLs.push(url);
  const anchor = $(id);
  anchor.href = url;
  anchor.download = name;
}
function clearExports() {
  for (const url of exportURLs) URL.revokeObjectURL(url);
  exportURLs = [];
  $("download").removeAttribute("href");
  $("csv").removeAttribute("href");
}
function prepareExports() {
  clearExports();
  exportLink(
    "download",
    segmentation,
    "mindglide-segmentation.nii",
    "application/octet-stream",
  );
  const csv =
    [
      "label,region,voxel_count,volume_mm3,volume_ml",
      ...labels.slice(1).map((name, i) => {
        const c = i + 1;
        return `${c},"${name}",${result.counts[c]},${result.volumes[c]},${result.volumes[c] / 1000}`;
      }),
    ].join("\n") + "\n";
  exportLink("csv", csv, "mindglide-volumes.csv", "text/csv;charset=utf-8");
}
$("boot").addEventListener("click", boot);
$("choose").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", () => choose($("file").files[0]));
$("example").addEventListener("click", () =>
  choose(new File([publicExample], "mni152.nii.gz")),
);
$("research").addEventListener("change", controls);
$("run").addEventListener("click", () => {
  busy = true;
  controls();
  worker.postMessage({ type: "segment" });
});
$("opacity").addEventListener("input", () =>
  nv?.setOpacity(1, Number($("opacity").value) / 100),
);
function clear() {
  ++generation;
  worker?.terminate();
  worker = null;
  clearVolumes();
  clearExports();
  if (original) new Uint8Array(original).fill(0);
  if (segmentation) segmentation.fill(0);
  original = segmentation = result = null;
}
$("clear").addEventListener("click", () => {
  clear();
  location.reload();
});
window.addEventListener("pagehide", clear);
controls();
