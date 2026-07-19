#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Idempotent vLLM XPU free-mem fallback patch (called from the build script).

Upstream vLLM (commit 91055efd3) already installs `get_mem_info_wrapper` in
`vllm/platforms/xpu.py` that routes `torch.accelerator.get_memory_info` ->
`_C_cache_ops.getMemoryInfo` (correct). If that op is present and working in the
installed vllm-xpu-kernels, NO patch is needed. This applies a one-line
`MemorySnapshot.measure()` fallback (use `torch.xpu.mem_get_info` on XPU) ONLY as
belt-and-suspenders for kernel builds where the wrapper still returns free=0.
Remove once verified sufficient on a compute node.

Usage: python patch_vllm_xpu_mem.py <path-to-vllm/utils/mem_utils.py>
"""
import sys

OLD = "        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)"
NEW = (
    "        # [ezpz] XPU mem_get_info fallback: upstream get_mem_info_wrapper\n"
    "        # (vllm/platforms/xpu.py) should make this redundant; kept for kernel\n"
    "        # builds where torch.accelerator.get_memory_info still returns free=0.\n"
    "        if getattr(device, 'type', None) == 'xpu':\n"
    "            self.free_memory, self.total_memory = torch.xpu.mem_get_info(device.index)\n"
    "        else:\n"
    "            self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_vllm_xpu_mem.py <mem_utils.py>")
        return 2
    path = sys.argv[1]
    try:
        s = open(path).read()
    except FileNotFoundError:
        print(f"  mem_utils.py not found at {path} -- skipping (nothing to patch)")
        return 0
    if "ezpz] XPU mem_get_info fallback" in s:
        print("  vllm mem_utils XPU fallback already present -- skipping")
        return 0
    if OLD not in s:
        print("  vllm mem_utils anchor not found (upstream changed) -- skipping")
        return 0
    open(path, "w").write(s.replace(OLD, NEW, 1))
    print("  patched vllm mem_utils XPU fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
