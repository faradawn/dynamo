<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Enable SGLang Hierarchical Cache (HiCache)

This guide shows how to enable SGLang's Hierarchical Cache (HiCache) inside Dynamo.

## 1) Build and start the SGLang container

```bash
./container/build.sh --framework sglang
./container/run.sh -it --framework sglang
```

## 2) Start the SGLang worker with HiCache enabled

```bash
python -m dynamo.sglang.worker \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --host 0.0.0.0 --port 8000 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-size 30 \
  --hicache-write-policy write_through \
  --hicache-storage-backend nixl \
  --log-level debug \
  --skip-tokenizer-init
```

- **--enable-hierarchical-cache**: Enables hierarchical KV cache/offload
- **--hicache-size**: HiCache capacity in GB of pinned host memory (upper bound of offloaded KV to CPU)
- **--hicache-write-policy**: Write policy (e.g., `write_through` for synchronous host writes)
- **--hicache-storage-backend**: Host storage backend for HiCache (e.g., `nixl`). NIXL selects the concrete store automatically; see [PR #8488](https://github.com/sgl-project/sglang/pull/8488)


Then, start the frontend:
```bash
python -m dynamo.frontend --http-port 8000
```

## 3) Send a test request

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
      {
        "role": "user",
        "content": "Explain why Roger Federer is considered one of the greatest tennis players of all time"
      }
    ],
    "stream": false,
    "max_tokens": 30
  }'
```

## 4) (Optional) Benchmarking

Run the perf script:
```bash
bash -x /workspace/benchmarks/llm/perf.sh \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --tensor-parallelism 1 \
  --data-parallelism 1 \
  --concurrency "2,4,8" \
  --input-sequence-length 2048 \
  --output-sequence-length 256
```
