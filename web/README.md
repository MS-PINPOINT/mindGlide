# MindGlide browser edition

A static, local browser interface to the public MindGlide model. Target domain: **https://mindglide.ms-pinpoint.com**. Academic MS-PINPOINT project; it is not Queen Square Analytics.

## Development

Use Node 22 or newer. `npm ci`, then `npm run prepare:assets` and `npm run dev`. `npm test` checks sliding-window geometry, bounded aggregation against a full accumulator, tie handling and invalid outputs. `npm run build` verifies the output is static and contains the pinned model and public example. `npm run preview` serves the production build with the same isolation and content-security headers.

The first asset preparation downloads public, checksum-pinned scientific wheels and runtime files. It does not need a Hugging Face, GitHub or Vercel secret. Generated model/runtime files, local examples, `.env*`, build output and validation inputs are ignored. Do not commit patient scans.

## Scientific contract

The HF checkpoint is pinned in `model-lock.json`. FP32, 20 logits including background, original 128 × 128 × 64 patches, 1 mm working spacing, half overlap, original Gaussian importance map. No quantisation, retraining or resolution reduction.

The five non-overlapping transposed convolutions are replaced with equivalent pointwise convolutions and spatial shuffles. `scripts/export_model.py` checks the permutation independently in float64, compares the full FP32 rewritten patch and ONNX output against PyTorch, and records artifact hashes. This accommodates ONNX Runtime Web's rank-5 ConvTranspose limitation.

`public/pipeline.py` runs NumPy/SciPy/scikit-image/NiBabel in Pyodide WebAssembly. It preserves RAS reorientation, positive-foreground cropping, reference cubic / anisotropic interpolation, nonzero intensity normalisation, output mask thresholds, lowest-label ties and 6-connected largest-component cleanup. Recovery and exports use the original scan grid. Unknown spatial units are treated as millimetres with a visible notice; other units are rejected.

`public/tiling.js` processes depth groups and reuses a 64-slice logits ring. A normal 256³ volume would otherwise need 1.25 GiB just for 20-class scores. Limits bound original voxels, decompressed bytes, resampled and padded voxels, anisotropic intermediate voxels and the accumulator. These limits are not a total peak-memory guarantee. `python tests/pipeline_bounds.py` checks rejection before expensive resizing in an isolated environment with NumPy/SciPy/scikit-image/NiBabel. Before file selection, the app warms the CPU backend and, where supported, the GPU backend on an original-size patch. Automatic mode selects the faster measured backend; CPU-only mode is also available. WebAssembly uses up to eight threads when isolation is supported. Processing is local and can take several minutes.

`scripts/validate_scientific.py` compares preprocessing and recovery with the existing MONAI tool on generated isotropic, flipped, permuted and anisotropic inputs, then compares full public-MNI PyTorch and native ONNX results. Reports are in `export-report.json`, `scientific-validation.json` and `browser-validation.json`. These checks demonstrate implementation agreement; they are not a clinical validation study or proof of performance on every scanner/device.

The public example is the complete TemplateFlow MNI152NLin2009cAsym 1 mm T1w template. It is served as `.nii-gz.bin` to preserve its compressed bytes across hosts that interpret a `.gz` extension as HTTP Content-Encoding. The file passed to the local NIfTI reader still has the normal `.nii.gz` name. Copyright and redistribution terms appear in `public/attribution.html`.

## Deployment and privacy

Deploy only `web` as an independent **MSPINPOINT** Vercel project, framework `null`, output `dist`. The main `MS-PINPOINT/mspinpoint` website links to it. This separation prevents the segmentation app from inheriting the main site's APIs, telemetry or environment secrets. Use `vercel deploy --scope mspinpoint`, never a QSA or personal team. Verify the project account ID before promotion.

The Vercel build fetches public artifacts using pinned SHA-256 hashes; source upload excludes generated runtime/model files, `.env*`, validation inputs and local deployment metadata. There are no API routes, Functions, middleware, proxies, databases, uploads, analytics, accounts or third-party runtime asset origins. There is no Function execution charge from segmentation. CDN transfer, requests, builds and optional firewall rate limiting are still metered hosting resources; static hosting is not an unlimited-cost guarantee.

COOP/COEP enable WebAssembly threads. The main document's CSP disallows eval and remote scripts. Only the dedicated processing worker allows JS eval because ONNX Runtime's generated WebAssembly bindings require `new Function`; it executes pinned engine code and does not evaluate user-supplied code. After complete runtime warmup, the worker disables fetch, XHR, importScripts, WebSocket, EventSource, Worker creation and WebTransport. The main app adds a CSP blocking all subsequent script, worker and network loading and disables its network APIs before enabling file selection. Viewer reads use local File streams, and exports use local blob downloads. No service worker, browser persistence or telemetry is used for MRI data.

Hosting request metadata (including IP addresses) still reaches Vercel. Public assets may be HTTP-cached. No non-essential cookies or tracking are used, so no cookie-consent banner is shown. The public notice is `public/privacy.html`; introduce prior consent and a reject/withdraw path if optional tracking is added later. Do not claim a host switch removes privacy obligations or that device memory can be forensically erased.

The deployed project's automatic DDoS mitigations remain enabled. Per-IP firewall limits restrict unusually frequent engine downloads and request floods; they can throttle shared-network users, and they do not guarantee a bill cap against distributed attacks. Any team-wide spend limit can pause the existing academic website as well, so review that consequence before changing it.

## Updating artifacts

Export in an isolated environment using the approved local GPU server if useful. Never serve that server or give it a public inference API. Compare results before changing any immutable model path or hash. Upload converted assets to the existing v1.3.0 GitHub release, with versioned names; attaching assets does not emit a new `release.published` event that would run the repository's package/container publication workflows. A new model requires a new asset filename and manifest version.

Run the browser with a public test scan, check overlay and exports, compare the exported labels with the stored native reference, inspect network activity after engine readiness, test clear/reset and malformed inputs, inspect production security headers, and verify deployment metadata has zero Functions before promoting. Browser/device support must remain accurately described on the page.
