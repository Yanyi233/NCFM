# Repository Guidelines
Review these notes before authoring or reviewing changes for the Neural Characteristic Function Matching (NCFM) project.

## Project Structure & Module Organization
Distillation orchestration lives in `condense/condense_script.py` and supporting logic inside `condenser/` (losses, sampling, evaluation helpers). `NCFM/` holds the neural characteristic function models, while `pretrain/` contains teacher training scripts and `evaluation/` wraps reproducibility runs. Cross-cutting utilities—distributed setup, augmentations, loaders—reside in `utils/`, and CLI argument handling sits in `argsprocessor/`. Configurations are grouped by images-per-class under `config/ipc*/{dataset}.yaml`. Stage raw inputs or distilled tensors in `data/` and `dataset/`, cache checkpoints under `pretrained_models/`, and write figures or logs to `results/` to keep the root clean.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`. Launch multi-GPU jobs through `torchrun`, for example `torchrun --nproc_per_node=4 pretrain/pretrain_script.py --gpu="0-3" --config_path=config/ipc50/cifar10.yaml`. Run condensation or evaluation from their respective directories so relative paths resolve: `cd condense && torchrun condense_script.py --ipc=50 --gpu="0,1" --config_path=../config/ipc50/cifar10.yaml`. Use `python test/test.py` to confirm PyTorch and CUDA availability, and `python extract_metrics.py --config config/ipc10/cifar100.yaml` when metrics tables need to be regenerated.

## Coding Style & Naming Conventions
Follow PEP 8 defaults: four-space indentation, lowercase module names, CamelCase classes such as `SampleNet`, and descriptive snake_case functions. Prefer f-strings for logging, keep distributed initialization inside `if __name__ == "__main__":`, and mirror config folder naming (`ipc10`, `ipc50`) when adding new YAML files. Check complex control flow with inline comments only where intent is not obvious.

## Testing Guidelines
A formal pytest suite is absent, so rely on targeted script checks. Validate data pipelines with `python test/data_check.py`, run a reduced `torchrun evaluation/evaluation_script.py --ipc=1 --gpu="0"` on new algorithms, and record resulting metrics under `results/` for comparison. Note GPU count, seed, and config path in merge discussions so reviewers can reproduce runs.

## Commit & Pull Request Guidelines
Commits in this repo stay short and purposeful (see `git log`); emulate that with single-sentence summaries—Chinese or English—describing the primary change. Keep unrelated edits in separate commits. Pull requests should summarize motivation, list validation commands, link issues or experiment trackers, and attach relevant log snippets or figures. Request review for algorithmic updates and wait for the documented commands to succeed before merging.

## Security & Configuration Tips
Do not commit secrets or licensed data—`test/test.py` shows how API keys stay commented or environment-backed. Scrub absolute paths from any new `config/` additions, and share download links for large checkpoints via `pretrained_models/README.md` instead of bundling binaries.
