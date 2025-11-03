# deep-learning-notebook Copilot Instructions

## Project Overview
- Single Jupyter notebook `deep-learning.ipynb` walks through MNIST classification, blending math exposition with executable cells.
- Core code builds a tiny autograd engine (`Scalar`, `Function`, `Module`, `MLP`) to illustrate gradients without external deps.
- Later sections pivot to practical training with PyTorch and Lightning, still inside the same notebook for side-by-side comparison.
- Data artifacts live under `data/MNIST`; torchvision downloads populate `raw/` automatically when loaders run.

## Coding Patterns
- Treat the custom autodiff stack as scalar-only: operations expect `Scalar` inputs, and network layers pass Python lists of `Scalar` objects.
- Keep utilities inline in the notebook; avoid creating new modules unless content must be reused elsewhere.
- Prefer minimal dependencies—reuse helpers already defined in earlier cells rather than importing new frameworks.
- Maintain deterministic behavior by respecting seeded RNG (`torch.manual_seed(42)`); extend seeding to Python `random` when adding training loops.

## Execution Workflow
- Run notebook top to bottom; later cells depend on classes defined above. Avoid reordering or clearing history without re-running everything.
- When adding training demos: first zero grads on custom params (`Module.zero_grad`), then propagate `Scalar.backward`, mirroring micrograd-style loops.
- Use existing MNIST download cells; they assume relative paths from repo root. Running elsewhere requires adjusting `root="data"`.
- Lightning workflow should integrate with `Trainer` from `lightning.pytorch`; keep callbacks lightweight (only `EarlyStopping` is imported today).

## Tooling & Environment
- Project targets Python `>=3.14` per `pyproject.toml`; if unavailable, rely on a newer interpreter (e.g., 3.12+) and adjust locally without editing manifest unless requested.
- Dependencies managed via `uv` (`uv.lock`) or PEP 517 builds; install with `pip install .` or `uv pip install -r pyproject.toml` equivalents.
- No automated tests exist; validation happens by executing notebook cells and comparing outputs to expectations stated in markdown.

## Contribution Tips
- Keep notebook outputs lightweight; strip massive tensors or training logs before committing unless explicitly needed.
- Reuse asset images under `assets/` by referencing relative paths to keep markdown portable.
- Document non-obvious training choices (batch size, learning rate) in adjacent markdown cells so readers follow the educational narrative.
- When extending functionality, favor separate notebook sections with clear headings to preserve the tutorial flow.
