"""Export the public, pinned MindGlide checkpoint without retraining.

The decoder's kernel == stride, padding == 0 transposed convolutions have
non-overlapping output blocks. A pointwise convolution and a spatial shuffle
are algebraically identical, and use operations supported by WebGPU.
The original network and this export are compared on the same input first.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from monai.data.utils import compute_importance_map

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'inference'))
from mindglide.network import get_network
from mindglide.consts import PATCH_SIZE
from mindglide.infer import HF_REPO_ID, HF_MODEL_FILENAME, HF_MODEL_REVISION


class SpatialShuffleUpsample(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.scale = tuple(source.stride)
        if (tuple(source.kernel_size) != self.scale or any(source.padding)
                or any(source.output_padding) or source.groups != 1
                or any(x != 1 for x in source.dilation)):
            raise ValueError('Unsupported transpose-convolution geometry')
        self.out_channels = source.out_channels
        self.pointwise = torch.nn.Conv3d(
            source.in_channels, source.out_channels * math.prod(self.scale), 1,
            bias=source.bias is not None,
        )
        with torch.no_grad():
            weights = source.weight.permute(1, 2, 3, 4, 0).contiguous()
            self.pointwise.weight.copy_(weights.reshape(self.pointwise.weight.shape))
            if source.bias is not None:
                self.pointwise.bias.copy_(source.bias.repeat_interleave(math.prod(self.scale)))

    def forward(self, x):
        n, _, a, b, c = x.shape
        sx, sy, sz = self.scale
        out = self.pointwise(x).reshape(n, self.out_channels, sx, sy, sz, a, b, c)
        return out.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
            n, self.out_channels, a * sx, b * sy, c * sz,
        )


def replace_upsampling(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.ConvTranspose3d):
            setattr(module, name, SpatialShuffleUpsample(child))
            count += 1
        else:
            count += replace_upsampling(child)
    return count


def file_info(path):
    return {'file': path.name, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(8)
    weights = hf_hub_download(HF_REPO_ID, HF_MODEL_FILENAME, revision=HF_MODEL_REVISION)
    original = get_network('cpu', checkpoint_path=weights).eval()
    # Independently verify the spatial permutation, without FP32 kernel noise.
    shuffle_errors = []
    for layer in original.modules():
        if isinstance(layer, torch.nn.ConvTranspose3d):
            source = copy.deepcopy(layer).double()
            shuffle = SpatialShuffleUpsample(source).double()
            # The constructor is FP32; copy exact double weights for this test.
            shuffle.pointwise.weight.data.copy_(source.weight.permute(1, 2, 3, 4, 0).reshape(shuffle.pointwise.weight.shape))
            if source.bias is not None:
                shuffle.pointwise.bias.data.copy_(source.bias.repeat_interleave(math.prod(source.stride)))
            trial = torch.randn(1, source.in_channels, 3, 2, 2, dtype=torch.float64)
            with torch.inference_mode():
                error = float((source(trial) - shuffle(trial)).abs().max())
            shuffle_errors.append(error)
            if error > 1e-12:
                raise RuntimeError('Spatial shuffle is not algebraically equivalent')
    rewritten = copy.deepcopy(original)
    replaced = replace_upsampling(rewritten)
    rewritten.eval()
    patch = torch.randn(1, 1, *PATCH_SIZE)
    began = time.perf_counter()
    with torch.inference_mode():
        reference = original(patch).numpy()
        rewritten_output = rewritten(patch).numpy()
    difference = np.abs(reference - rewritten_output)
    agreement = float(np.mean(reference.argmax(1) == rewritten_output.argmax(1)))
    if float(difference.max()) > 1e-3 or agreement < 0.9999:
        raise RuntimeError(f'Rewrite differs: max absolute error {difference.max()}')
    print(json.dumps({'rewrite_count': replaced, 'rewrite_max_abs': float(difference.max()),
                      'rewrite_label_agreement': agreement,
                      'reference_seconds': time.perf_counter() - began}), flush=True)
    target = args.out / 'mindglide-fp32.onnx'
    torch.onnx.export(rewritten, patch, target, input_names=['image'],
                      output_names=['logits'], opset_version=18, dynamo=False,
                      do_constant_folding=True)
    graph = onnx.load(target)
    # Publish no exporter filesystem paths or debug annotations.
    graph.doc_string = ''
    graph.graph.doc_string = ''
    for node in graph.graph.node:
        node.doc_string = ''
    onnx.save(graph, target)
    onnx.checker.check_model(graph)
    unsupported = [n.op_type for n in graph.graph.node if n.op_type == 'ConvTranspose']
    if unsupported:
        raise RuntimeError('Export retained ConvTranspose')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'],
                                  sess_options=options)
    output = session.run(None, {'image': patch.numpy()})[0]
    onnx_error = float(np.max(np.abs(output - reference)))
    onnx_agreement = float(np.mean(output.argmax(1) == reference.argmax(1)))
    if not np.allclose(output, reference, atol=2e-3, rtol=1e-3) or onnx_agreement < 0.9999:
        raise RuntimeError(f'ONNX patch differs: max error {onnx_error}')
    importance = compute_importance_map(PATCH_SIZE, mode='gaussian', sigma_scale=0.125).numpy()
    importance_path = args.out / 'gaussian-f32.bin'
    importance.astype('<f4').tofile(importance_path)
    report = {'source': {'repo': HF_REPO_ID, 'revision': HF_MODEL_REVISION,
                         'checkpoint': HF_MODEL_FILENAME},
              'patch': PATCH_SIZE, 'classes': 20, 'overlap': 0.5,
              'precision': 'float32', 'opset': 18, 'rewrite_count': replaced,
              'shuffle_float64_max_abs': max(shuffle_errors),
              'rewrite_max_abs': float(difference.max()),
              'rewrite_label_agreement': agreement,
              'onnx_max_abs': onnx_error, 'onnx_label_agreement': onnx_agreement,
              'operators': sorted({n.op_type for n in graph.graph.node}),
              'model': file_info(target), 'importance': file_info(importance_path)}
    (args.out / 'export-report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.out / 'patch-reference.npz', input=patch.numpy(), logits=reference)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
