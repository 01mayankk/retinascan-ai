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
| Model training | Training lives in notebooks; tracked Python files under `scripts/training/` are empty placeholders. |
| Dataset provenance | Incomplete: scripts write `train/val/test`, while training notebooks use `enhanced_*`; no checked-in link establishes their lineage. |

## Evaluation outputs in notebooks

Stored notebook outputs report:

- EfficientNet-B2 with test-time augmentation: **68.89%** test accuracy.
- DenseNet121 phase-2 checkpoint: **68.00%** test accuracy.
- Weighted ensemble (DenseNet121 0.6, EfficientNet-B2 0.4): **70.22%** test accuracy.

These are saved notebook results on notebook `enhanced_test` paths. They were not reproduced during this documentation work, and the repository does not establish that the notebook dataset matches the output of the checked-in preprocessing scripts. They are not a clinical validation claim or a guarantee of current deployed performance.

## Known limitations

- Preprocessing augments minority-class data before splitting; file-wise splitting does not keep source-image variants together.
- The random split and balancing process have no fixed seed.
- The raw dataset is not included, so original class counts and the exact distribution of a particular run are unavailable.
- The training notebooks use `enhanced_*` paths with no verified generation pipeline in this repository.
- The runtime pipeline does not include quality filtering, CLAHE, circular cropping, or Ben Graham preprocessing.
- The app loads the DenseNet121 phase-2 checkpoint; notebook phase-3 settings/results do not describe the checkpoint currently loaded by the app.
- The backend writes input and heatmap to fixed `latest_*` paths.
- The API returns report text and files, not a full per-class probability vector.
- Grad-CAM visualizes model attribution, not confirmed retinal lesions.
- Model outputs are not a diagnosis and require licensed ophthalmologist review.

## Suggested roadmap

1. Record dataset provenance, class counts, subject/source identifiers, and immutable split manifests.
2. Split source images before training-only augmentation; add deterministic seeds and group-aware leakage checks.
3. Make the `enhanced_*` dataset construction reproducible or remove the ambiguous path distinction.
4. Re-run evaluation from a documented, held-out dataset and publish per-class metrics and uncertainty/calibration analysis.
5. Add runtime request isolation for generated files if concurrent serving is required.
6. Conduct appropriate clinical, privacy, and regulatory review before any real-world medical use.
