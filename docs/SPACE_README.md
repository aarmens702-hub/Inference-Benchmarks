# HuggingFace Space README

When you create the Space, paste the YAML block below at the very top
of the Space's `README.md`. The Space repo is separate from this GitHub
repo (or a separate branch — see `Makefile`).

```
---
title: InferBench
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# InferBench live demo

FP32 vs INT8 DistilBERT-SST2, side by side. Type a sentence, see both
predictions and both latency numbers.

Full project: https://github.com/aarmens702-hub/Inference-Benchmarks
```

After creating the Space:

```
make space-remote                # one-time: add the huggingface git remote
make deploy-space                # push main; the Space builds Dockerfile.spaces
```

Override `HF_SPACE` if the URL differs:

```
make HF_SPACE=huggingface.co/spaces/yourname/yourspace deploy-space
```

If you want the GitHub repo and the Space repo to share the same
history, the Space repo can be a sibling git remote of this repo. If
you prefer them separate, push only a `space` branch:

```
git checkout -b space
git push $(HF_REMOTE_NAME) space:main
```
