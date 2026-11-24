# Architecture decisions and implementation records

These records describe choices observable in checked-in code. The repository does not preserve the original design discussions; where author intent is unavailable, that is stated instead of inferred as historical fact.

## ADR-001: Use EfficientNet-B2 and DenseNet121 in the runtime ensemble

- **Status:** Implemented.
- **Evidence:** `app/inference.py` loads both checkpoints, computes separate softmax probability tensors, and combines them with DenseNet121 weight 0.6 and EfficientNet-B2 weight 0.4.
- **Operational rationale visible in code:** The serving prediction is produced from a weighted combination of two model outputs.
- **Original model-selection rationale:** Not recorded in the checked-in source.
- **Implication:** Both checkpoint files are required for local inference. Their notebook-reported evaluation scores are historical and depend on the notebook test data.

## ADR-002: Balance classes using sampling and augmentation before splitting

- **Status:** Implemented in `scripts/preprocessing/`.
- **Evidence:** `balance_dataset.py` targets 300 files per class, randomly undersamples classes at or above the target, and augments smaller classes with horizontal flips, rotations, and brightness/contrast changes. `split_dataset.py` then performs a per-class 70/15/15 file split.
- **Original rationale:** Not recorded in the source.
- **Known tradeoff:** Pre-split augmentation and file-wise random splitting can place related variants in different splits. No source-image grouping, fixed seed, or class-weighted loss is configured in these scripts/notebook active training setup.
- **Follow-up:** Preserve source-image identifiers and split originals before augmenting training-only data; record a reproducible manifest and seed.

## ADR-003: Fine-tune convolutional image-classification backbones

- **Status:** Implemented in notebooks 05 and 07.
- **Evidence:** Notebook 05 uses pretrained torchvision EfficientNet-B2 weights and unfreezes its final two feature blocks. Notebook 07 defines staged DenseNet121 training; the application loads the phase-2 DenseNet checkpoint.
- **Training safeguards present:** Dropout is configured in both classifier heads; validation-loss scheduling and early stopping are used. DenseNet phase 3 adds label smoothing and weight decay, although the app does not load that phase's final checkpoint.
- **Original backbone-selection rationale:** Not recorded in the repository.
- **Implication:** The app's DenseNet checkpoint and the final DenseNet notebook checkpoint are not interchangeable without updating the architecture/checkpoint selection deliberately.

## ADR-004: Generate Grad-CAM attribution for the DenseNet model

- **Status:** Implemented.
- **Evidence:** `app/gradcam.py` selects `densenet_model.features[-1]`, generates an overlay, and saves it as `outputs/heatmaps/latest_gradcam.png`.
- **Original choice rationale or comparison with other XAI methods:** Not recorded in the source.
- **Implication:** The visualization is model attribution. It is not a segmentation, a clinically confirmed lesion, or evidence that a specific pathology is present. The `predicted_class` parameter is accepted but not passed as an explicit CAM target.

## ADR-005: Separate the static frontend from the Gradio inference service

- **Status:** Implemented.
- **Evidence:** `frontend/` builds as a Vite static site; root `vercel.json` publishes `frontend/dist`; `@gradio/client` connects to the public `01mayankk/RetinaScanAI` Space.
- **Security property:** The frontend requires no HF token for the public Space. `VITE_*` values are public configuration and must not contain secrets.
- **Original hosting rationale:** Not recorded in the source.
- **Implication:** Frontend and backend are deployed independently. The frontend endpoint setting must match the Gradio API name exposed by its target Space (`/run_analysis` in the checked-in frontend example; `/predict` for the local app).

## ADR-006: Separate clinical guidance from technical pipeline inspection

- **Status:** Implemented.
- **Evidence:** `frontend/index.html` retains the grading guide and clinical disclaimer, while pipeline details are inspected with `python scripts/pipeline_summary.py`.
- **Rationale:** Keep the public screening workflow focused on clinical end users while making code-derived methodology available to developers and reviewers.
- **Implication:** The CLI is read-only and reports source wiring separately from local data availability; historical notebook accuracies remain stored outputs rather than product guarantees.
