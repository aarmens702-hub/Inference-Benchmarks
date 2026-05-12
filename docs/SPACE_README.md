# HuggingFace Spaces deploy

The main `README.md` already carries the HF Spaces YAML frontmatter at the
top (title, emoji, sdk=docker, app_port=7860). The Space picks it up
automatically when you push.

Build target: the root `Dockerfile` (same one used by `docker compose`).
It bakes the FP32 + INT8 ONNX models into the image at build time so the
Space boots ready to serve.

## First-time setup

1. Create the Space at https://huggingface.co/new-space with SDK = **Docker**, template = **Blank**.

2. Add an HF access token with **write** scope at https://huggingface.co/settings/tokens.

3. Save the token to your local credential store:

       pip install huggingface_hub
       hf auth login            # paste the token when prompted

4. Add the Space as a git remote and push:

       make space-remote        # one-time: git remote add space https://huggingface.co/spaces/Aarmen/inferbench
       make deploy-space        # git push space main

If your Space lives at a different URL, override:

    make HF_SPACE=huggingface.co/spaces/<owner>/<name> deploy-space

## Subsequent deploys

    git push                     # to github
    make deploy-space            # to huggingface

Or push to both remotes at once:

    git push origin main && git push space main

## Build behaviour

The first build runs `python -m inferbench.models.export_model` and
`python -m inferbench.models.quantize_model` inside the image, which
takes ~3–5 minutes. HF caches the Docker layer, so subsequent code-only
pushes reuse the model layer and rebuild in ~30 seconds.
