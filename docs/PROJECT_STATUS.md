# Project status

## Current implementation

| Area | Status |
|---|---|
| Upload UI | Implemented for JPG/PNG file selection, drag-and-drop, preview, and image dimensions. |
| Frontend API client | Implemented with `@gradio/client`, queue/loading states, and a public Space default. |
| Structured result rendering | Implemented for prediction, confidence, runner-up class, risk, confidence status, and recommendation; raw report is collapsible. |
| Explainability | Original image and Grad-CAM heatmap views are available. The runtime Grad-CAM uses DenseNet121. |
| PDF | Backend generates a timestamped PDF with result data, original image, heatmap, and disclaimer. |
| Frontend deployment | Vercel build configuration is present; no verified RetinaScan AI deployment URL is recorded. The candidate `retinascan-ai.vercel.app` serves an unrelated Next.js starter. |
| Data preparation | Label copy, balancing, splitting, and image counting scripts are present. |
| Pipeline inspection | Read-only `python scripts/pipeline_summary.py` CLI reports source mappings and local data/model artifact availability. |
| Model training | Training lives in notebooks; tracked Python files under `scripts/training/` are empty placeholders. |
| Dataset provenance | Source mapping is present: notebook 04 writes the `enhanced_*` paths loaded by notebooks 05 and 07. Dataset files are absent locally, so actual current counts are unavailable. |

## Evaluation outputs in notebooks

Stored notebook outputs report:

- EfficientNet-B2 with test-time augmentation: **68.89%** test accuracy.
- DenseNet121 phase-2 checkpoint: **68.00%** test accuracy.
- Weighted ensemble (DenseNet121 0.6, EfficientNet-B2 0.4): **70.22%** test accuracy.

These are saved notebook results on notebook `enhanced_test` paths. They were not reproduced during this documentation work. Notebook 04 connects the scripted splits to the enhanced paths, but no current local dataset is available to independently verify the data or reproduce these results. They are not a clinical validation claim or a guarantee of current deployed performance.

## Known limitations

- Preprocessing augments minority-class data before splitting; file-wise splitting does not keep source-image variants together.
- The random split and balancing process have no fixed seed.
- The raw dataset is not included, so original class counts and the exact distribution of a particular run are unavailable.
- Actual dataset files and per-class counts are unavailable in this workspace; the verified source mapping does not independently prove a specific historical run's contents.
- The runtime pipeline does not include quality filtering, CLAHE, circular cropping, or Ben Graham preprocessing.
- The app loads the DenseNet121 phase-2 checkpoint; notebook phase-3 settings/results do not describe the checkpoint currently loaded by the app.
- The backend writes input and heatmap to fixed `latest_*` paths.
- The API returns report text and files, not a full per-class probability vector.
- Grad-CAM visualizes model attribution, not confirmed retinal lesions.
- Model outputs are not a diagnosis and require licensed ophthalmologist review.

## Suggested roadmap

1. Record dataset provenance, class counts, subject/source identifiers, and immutable split manifests.
2. Split source images before training-only augmentation; add deterministic seeds and group-aware leakage checks.
3. Record immutable manifests and source-image identities for each generated `enhanced_*` dataset.
4. Re-run evaluation from a documented, held-out dataset and publish per-class metrics and uncertainty/calibration analysis.
5. Add runtime request isolation for generated files if concurrent serving is required.
6. Conduct appropriate clinical, privacy, and regulatory review before any real-world medical use.
