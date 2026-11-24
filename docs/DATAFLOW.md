# Dataflow

## Offline dataset preparation and training

```mermaid
flowchart LR
    CSV[train.csv id_code diagnosis] --> MAP[classify_images.py]
    RAW[raw_images id_code.png] --> MAP
    MAP --> CL[classified_images 0 to 4]
    CL --> BAL[balance_dataset.py target 300 per class]
    BAL --> AUG[Random undersampling or augmented copies]
    AUG --> BD[balanced_dataset]
    BD --> SPLIT[split_dataset.py class-wise shuffle]
    SPLIT --> TR[train 70 percent]
    SPLIT --> VA[val 15 percent]
    SPLIT --> TE[test remainder]
    TR --> ENHANCE[notebook 04 CLAHE and conditional sharpening]
    VA --> ENHANCE
    TE --> ENHANCE
    ENHANCE --> ENH[enhanced_train enhanced_val enhanced_test]
    ENH --> TRAIN[Training notebooks 05 and 07]
    TRAIN --> CKPT[Model checkpoints]
    CKPT --> SERVE[app.inference]
```

`classify_images.py` copies `<id_code>.png` to the numeric directory matching `diagnosis`. `balance_dataset.py` randomly samples larger classes down to 300 or creates augmented copies for smaller classes. `split_dataset.py` shuffles each balanced class independently and copies integer-truncated 70% / 15% / remainder slices.

Notebook 04 maps each script-generated split to its matching `enhanced_*` directory, applies `enhance_retina_image`, and saves the processed images. Notebooks 05 and 07 load those enhanced outputs. This verifies source wiring, not the contents of a current local dataset: the dataset is not included in the repository.

Because augmentation is performed before the per-file split and there is no grouping by original source file, related variants could be distributed across training and evaluation. No fixed random seed or explicit source-group leakage prevention is present. If every class has exactly 300 files, the intended totals are 1,050 train, 225 validation, and 225 test.

Run `python scripts/pipeline_summary.py` from the repository root to inspect these source mappings, local counts when data folders exist, notebook safeguards, and model artifact metadata. The CLI is read-only. Notebook 04's CLAHE and conditional sharpening are offline preprocessing and are not called during live inference.

## Real-time screening

```mermaid
sequenceDiagram
    actor User
    participant UI as Vite frontend
    participant GC as @gradio/client
    participant API as Gradio Space app.py
    participant INF as app.inference
    participant CAM as app.gradcam
    participant PDF as app.report_generator
    User->>UI: Select JPG or PNG
    UI->>UI: Validate type, preview, read dimensions
    User->>UI: Start screening
    UI->>GC: Connect to configured Space
    GC->>API: Queue image request
    API->>INF: Convert RGB, resize 300x300, normalize
    INF->>INF: EfficientNet-B2 and DenseNet121 forward passes
    INF->>INF: Weighted softmax sum 0.4 and 0.6
    INF-->>API: Top two classes, confidence, risk, recommendation
    API->>CAM: Generate DenseNet121 Grad-CAM overlay
    CAM-->>API: Heatmap path
    API->>PDF: Build report PDF
    PDF-->>API: PDF path
    API-->>GC: Report text, heatmap file, PDF file
    GC-->>UI: Queue status and data events
    UI->>UI: Parse report fields and render summary
    UI-->>User: Summary, attribution view, PDF download
```

The frontend renders an original-image view and a Grad-CAM overlay view, parses report text into structured fields, and retains the original response behind a raw-output disclosure. The Grad-CAM image is an attribution visualization and must not be interpreted as a confirmed lesion map.

## Runtime image transforms

The serving path in `app/inference.py` performs RGB conversion, resize to 300 × 300, tensor conversion, and normalization with ImageNet channel means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. The inference path does not include CLAHE, a circular crop, Ben Graham preprocessing, sharpening, or corrupt-image filtering.
