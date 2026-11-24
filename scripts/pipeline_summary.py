"""Print a read-only, source-backed summary of the RetinaScan AI pipeline."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = ROOT / "scripts" / "preprocessing"
NOTEBOOKS = ROOT / "notebooks"
MODELS = ROOT / "models"
IMAGE_FILTER_RE = re.compile(r"(?:requires_grad|features\[|denseblock|transition|Dropout|label_smoothing|weight_decay|patience_counter|best_val|random\.seed|random\.shuffle)", re.I)
TRANSFORM_NAMES = {
    "Resize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "RandomAffine",
    "ColorJitter",
    "GaussianBlur",
    "Normalize",
    "ToTensor",
}


def read_python(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def target_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for item in target.elts:
            names.extend(target_names(item))
        return names
    return []


def top_level_assignments(tree):
    assignments = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                for name in target_names(target):
                    assignments.append((name, statement.value))
        elif isinstance(statement, ast.AnnAssign):
            for name in target_names(statement.target):
                assignments.append((name, statement.value))
    return assignments


def literal_values(path):
    values = {}
    for name, value in top_level_assignments(read_python(path)):
        try:
            values[name] = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
    return values


def expression(node):
    return ast.unparse(node)


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def call_expressions(tree, predicate):
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and predicate(call_name(node.func))
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    return list(dict.fromkeys(expression(node) for node in calls))


def notebook_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    skipped = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        try:
            tree = ast.parse(source, filename=f"{path.name}:cell-{index}")
        except SyntaxError as error:
            if source.strip():
                skipped.append((index, error.msg))
            continue
        cells.append((index, cell.get("execution_count"), source, tree))
    return cells, skipped


def notebook_assignments(cells, wanted):
    found = {name: [] for name in wanted}
    for index, execution_count, _source, tree in cells:
        for name, value in top_level_assignments(tree):
            if name in found:
                found[name].append((index, execution_count, value))
    return found


def resolve_source_path(source_file, relative_path):
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (source_file.parent / candidate).resolve()


def image_suffixes(path):
    tree = read_python(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "endswith" or not node.args:
            continue
        try:
            suffixes = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if isinstance(suffixes, tuple) and all(isinstance(item, str) for item in suffixes):
            return suffixes
    raise ValueError(f"No literal image suffix filter found in {path}")


def count_image_folders(directory, suffixes, class_names):
    if not directory.is_dir():
        print(f"  {directory.relative_to(ROOT)}: not present locally")
        return None

    total = 0
    folders = sorted(item for item in directory.iterdir() if item.is_dir())
    if not folders:
        print(f"  {directory.relative_to(ROOT)}: no class folders found")
        return 0

    for folder in folders:
        count = sum(
            1 for item in folder.iterdir()
            if item.name.endswith(suffixes)
        )
        total += count
        try:
            label = class_names.get(int(folder.name), "unmapped")
        except ValueError:
            label = "unmapped"
        print(f"  {folder.name} ({label}): {count}")
    print(f"  Total: {total}")
    return total


def print_notebook_summary(path, constants):
    cells, skipped = notebook_cells(path)
    wanted = {
        "train_dir",
        "val_dir",
        "test_dir",
        "IMAGE_SIZE",
        "BATCH_SIZE",
        "NUM_EPOCHS",
        "EARLY_STOPPING_PATIENCE",
        "EPOCHS",
        "PATIENCE",
        "PHASE1_LR",
        "PHASE2_LR",
        "PHASE3_LR",
        "PHASE1_EPOCHS",
        "PHASE2_EPOCHS",
        "PHASE3_EPOCHS",
        "PHASE1_MODEL_PATH",
        "PHASE2_MODEL_PATH",
        "FINAL_MODEL_PATH",
        "criterion",
        "optimizer",
        "scheduler",
    }
    assignments = notebook_assignments(cells, wanted)
    print(f"\n  {path.relative_to(ROOT)}")

    for name in ("train_dir", "val_dir", "test_dir"):
        entries = assignments[name]
        if entries:
            index, _execution_count, value = entries[-1]
            print(f"    {name} (cell {index}): {expression(value)}")

    settings = []
    for name in constants:
        entries = assignments.get(name, [])
        if entries:
            values = ", ".join(expression(value) for _cell, _count, value in entries)
            settings.append(f"{name}={values}")
    if settings:
        print(f"    settings: {'; '.join(settings)}")

    for name in ("PHASE1_MODEL_PATH", "PHASE2_MODEL_PATH", "FINAL_MODEL_PATH"):
        entries = assignments.get(name, [])
        if entries:
            values = ", ".join(expression(value) for _cell, _count, value in entries)
            print(f"    {name}: {values}")

    trees = [tree for _index, _count, _source, tree in cells]
    backbones = []
    dropout = []
    transforms = []
    for tree in trees:
        backbones.extend(
            call_expressions(
                tree,
                lambda name: name in {"models.efficientnet_b2", "models.densenet121"},
            )
        )
        dropout.extend(
            call_expressions(tree, lambda name: name == "nn.Dropout")
        )
        transforms.extend(
            call_expressions(
                tree,
                lambda name: name.startswith("transforms.")
                and name.rsplit(".", 1)[-1] in TRANSFORM_NAMES,
            )
        )
    if backbones:
        print(f"    model constructors: {'; '.join(dict.fromkeys(backbones))}")
    if dropout:
        print(f"    dropout: {'; '.join(dict.fromkeys(dropout))}")
    if transforms:
        print(f"    transforms: {'; '.join(dict.fromkeys(transforms))}")

    for name in ("criterion", "optimizer", "scheduler"):
        entries = assignments[name]
        if entries:
            values = "; ".join(
                f"cell {index}: {name} = {expression(value)}"
                for index, _count, value in entries
            )
            print(f"    {name} assignments (source order): {values}")
    criterion_entries = assignments["criterion"]
    if path.name == "05_efficientnet_training.ipynb" and criterion_entries:
        final_criterion = expression(criterion_entries[-1][2])
        print(f"    effective criterion by source order: {final_criterion}")
        if (
            len(criterion_entries) > 1
            and "FocalLoss" in expression(criterion_entries[0][2])
            and "CrossEntropyLoss" in final_criterion
        ):
            print("    note: the later CrossEntropyLoss assignment replaces the earlier FocalLoss setup.")
    elif path.name == "07_densenet121_training.ipynb":
        print(
            "    phase note: phase 1/2 use CrossEntropyLoss; phase 3 later sets "
            "label_smoothing=0.1. app/inference.py loads the phase-2 checkpoint."
        )

    safeguard_lines = []
    for index, _count, source, _tree in cells:
        for line in source.splitlines():
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and IMAGE_EVIDENCE_RE.search(stripped)
            ):
                safeguard_lines.append(f"cell {index}: {stripped}")
    if safeguard_lines:
        print("    freeze/unfreeze and regularization source:")
        for line in dict.fromkeys(safeguard_lines):
            print(f"      {line}")

    if skipped:
        print(f"    AST-skipped code cells: {len(skipped)} (notebook magics or unsupported syntax)")
    return cells, skipped, assignments


def print_model_inventory(config_values, inference_values):
    print("\n4. Local model inventory and runtime inference")
    expected = []
    for values in (config_values, inference_values):
        for key in ("EFFICIENTNET_MODEL_PATH", "DENSENET_MODEL_PATH"):
            path = values.get(key)
            if path and path not in expected:
                expected.append(path)

    if not MODELS.is_dir():
        print(f"  Model directory not present: {MODELS.relative_to(ROOT)}")
        return

    files = sorted(path for path in MODELS.rglob("*") if path.is_file())
    if not files:
        print("  No files found in models/.")
    for path in files:
        size_mib = path.stat().st_size / (1024 * 1024)
        print(f"  {path.relative_to(ROOT)}: {size_mib:.2f} MiB (file metadata only)")

    for name in expected:
        path = (ROOT / name).resolve()
        state = "present" if path.is_file() else "missing"
        print(f"  configured checkpoint {name}: {state}")

    weights = literal_values(ROOT / "app" / "inference.py")
    config_weights = literal_values(ROOT / "app" / "config.py")
    print(
        "  ensemble weights from app/inference.py: "
        f"DenseNet121={weights.get('DENSENET_WEIGHT')}, "
        f"EfficientNet-B2={weights.get('EFFICIENTNET_WEIGHT')}"
    )
    print(
        "  input size / confidence threshold from app/config.py: "
        f"{config_values.get('IMAGE_SIZE')} px / "
        f"{config_values.get('CONFIDENCE_THRESHOLD')}%"
    )
    transform_calls = call_expressions(
        read_python(ROOT / "app" / "inference.py"),
        lambda name: name in {
            "transforms.Resize",
            "transforms.ToTensor",
            "transforms.Normalize",
        },
    )
    print(f"  runtime preprocessing: {'; '.join(transform_calls)}")
    inference_tree = read_python(ROOT / "app" / "inference.py")
    model_calls = call_expressions(
        inference_tree,
        lambda name: name in {"models.efficientnet_b2", "models.densenet121"},
    )
    dropout_calls = call_expressions(
        inference_tree,
        lambda name: name == "nn.Dropout",
    )
    print(f"  runtime model constructors: {'; '.join(model_calls)}")
    print(f"  runtime classifier dropout: {'; '.join(dropout_calls)}")
    compared_keys = (
        "EFFICIENTNET_MODEL_PATH",
        "DENSENET_MODEL_PATH",
        "DENSENET_WEIGHT",
        "EFFICIENTNET_WEIGHT",
    )
    if any(weights.get(key) != config_weights.get(key) for key in compared_keys):
        print("  NOTE: inference.py and config.py constants differ; runtime values above come from inference.py.")


IMAGE_EVIDENCE_RE = re.compile(
    r"requires_grad|features\[|denseblock|transition|Dropout|label_smoothing|"
    r"weight_decay|patience_counter|best_val|ReduceLROnPlateau|"
    r"random\.seed|random\.shuffle",
    re.I,
)


def main():
    classify_path = PREPROCESSING / "classify_images.py"
    balance_path = PREPROCESSING / "balance_dataset.py"
    split_path = PREPROCESSING / "split_dataset.py"
    count_classified_path = PREPROCESSING / "count_images.py"
    count_balanced_path = PREPROCESSING / "count_balanced_dataset.py"
    config_path = ROOT / "app" / "config.py"
    inference_path = ROOT / "app" / "inference.py"
    quality_notebook_path = NOTEBOOKS / "04_image_quality_enhancement.ipynb"
    efficientnet_path = NOTEBOOKS / "05_efficientnet_training.ipynb"
    densenet_path = NOTEBOOKS / "07_densenet121_training.ipynb"

    config_values = literal_values(config_path)
    class_names = config_values["CLASS_NAMES"]
    classify_values = literal_values(classify_path)
    balance_values = literal_values(balance_path)
    split_values = literal_values(split_path)
    count_classified_values = literal_values(count_classified_path)
    count_balanced_values = literal_values(count_balanced_path)

    print("RetinaScan AI pipeline inspection (read-only)")
    print("Source files are parsed; dataset counts and model sizes are read from local paths.")

    print("\n1. Raw data ingestion and class mapping")
    print(f"  CSV source: {classify_values['csv_file']}")
    print(f"  Raw image folder: {classify_values['image_folder']}")
    print(f"  Classified output: {classify_values['output_folder']}")
    print("  Rule: row['id_code'] + '.png'; folder = str(row['diagnosis']); copy if source exists.")
    print("  Actual classifier folders are numeric; labels below are from app/config.py:")
    for index, label in sorted(class_names.items()):
        folder = label.replace(" ", "_")
        print(f"    {index} -> {label} (display name: {index}_{folder})")
    print("  The classifier script does not validate image contents.")

    print("\n2. Class imbalance, augmentation, and source counts")
    print(
        f"  Balance input/output: {balance_values['input_base']} -> "
        f"{balance_values['output_base']}"
    )
    print(f"  Target per class: {balance_values['TARGET_COUNT']}")
    print("  Transform calls from balance_dataset.py:")
    for call in call_expressions(
        read_python(balance_path),
        lambda name: name.startswith("transforms.")
        and name.rsplit(".", 1)[-1] in TRANSFORM_NAMES,
    ):
        print(f"    {call}")
    print("  Branch: >= target uses random.sample undersampling; below target copies originals and augments random choices.")
    balance_tree = read_python(balance_path)
    if not any(
        isinstance(node, ast.Call) and call_name(node.func) == "random.seed"
        for node in ast.walk(balance_tree)
    ):
        print("  Random seed: none configured in balance_dataset.py.")

    classified_dir = resolve_source_path(
        count_classified_path, count_classified_values["base_folder"]
    )
    balanced_dir = resolve_source_path(
        count_balanced_path, count_balanced_values["base_folder"]
    )
    classified_suffixes = image_suffixes(count_classified_path)
    balanced_suffixes = image_suffixes(count_balanced_path)
    print(
        f"  count_images.py counts case-sensitive suffixes "
        f"{classified_suffixes} under {classified_dir.relative_to(ROOT)}:"
    )
    count_image_folders(classified_dir, classified_suffixes, class_names)
    print(
        f"  count_balanced_dataset.py counts case-sensitive suffixes "
        f"{balanced_suffixes} under {balanced_dir.relative_to(ROOT)}:"
    )
    count_image_folders(balanced_dir, balanced_suffixes, class_names)

    print("\n3. Split ratios and verified notebook lineage")
    print(f"  Split input: {split_values['source_dir']}")
    print(
        "  Ratios: "
        f"train={split_values['train_ratio']:.0%}, "
        f"validation={split_values['val_ratio']:.0%}, "
        f"test={split_values['test_ratio']:.0%} (remainder after integer truncation)"
    )
    split_paths = {
        name: resolve_source_path(split_path, split_values[name])
        for name in ("train_dir", "val_dir", "test_dir")
    }
    print("  Split implementation: random.shuffle per class; no fixed seed or source-group split.")
    split_suffixes = image_suffixes(split_path)
    for name, path in split_paths.items():
        print(f"  Local {name} counts ({path.relative_to(ROOT)}):")
        count_image_folders(path, split_suffixes, class_names)

    quality_cells, quality_skipped = notebook_cells(quality_notebook_path)
    quality_assignments = notebook_assignments(quality_cells, {"dataset_pairs"})
    pair_entries = quality_assignments["dataset_pairs"]
    pairs = ast.literal_eval(pair_entries[-1][2]) if pair_entries else []
    efficient_cells, efficient_skipped = notebook_cells(efficientnet_path)
    dense_cells, dense_skipped = notebook_cells(densenet_path)
    efficient_dirs = notebook_assignments(
        efficient_cells, {"train_dir", "val_dir", "test_dir"}
    )
    dense_dirs = notebook_assignments(
        dense_cells, {"train_dir", "val_dir", "test_dir"}
    )
    pair_sources = {}
    pair_targets = {}
    if isinstance(pairs, list):
        for source, target in pairs:
            pair_sources[Path(source).name] = resolve_source_path(
                quality_notebook_path, source
            )
            pair_targets[Path(target).name] = resolve_source_path(
                quality_notebook_path, target
            )

    quality_source = "\n".join(source for _index, _count, source, _tree in quality_cells)
    has_enhancement_flow = (
        "def enhance_retina_image" in quality_source
        and "enhance_retina_image(" in quality_source
        and "cv2.imwrite(" in quality_source
    )
    split_matches = all(
        pair_sources.get(name.removesuffix("_dir")) == path
        for name, path in split_paths.items()
    )
    training_matches = True
    for notebook_path, paths in (
        (efficientnet_path, efficient_dirs),
        (densenet_path, dense_dirs),
    ):
        for name in ("train_dir", "val_dir", "test_dir"):
            entries = paths[name]
            if not entries:
                training_matches = False
                continue
            raw_path = ast.literal_eval(entries[-1][2])
            if resolve_source_path(notebook_path, raw_path) not in pair_targets.values():
                training_matches = False
    lineage_verified = bool(pairs) and has_enhancement_flow and split_matches and training_matches
    for source, target in pairs:
        print(f"  Notebook 04 pair: {source} -> {target}")
    if lineage_verified:
        print(
            "  VERIFIED SOURCE LINEAGE: split_dataset.py outputs train/val/test; "
            "notebook 04 processes those pairs with enhance_retina_image and cv2.imwrite; "
            "notebooks 05 and 07 load the matching enhanced_* outputs."
        )
    else:
        print(
            "  SOURCE LINEAGE NOT FULLY VERIFIED: the expected split, enhancement, "
            "and training path references did not all match."
        )
    print("  This verifies source wiring only; local data folders are reported separately above.")

    print("\n  Training notebook safeguards and model configuration (source order):")
    print_notebook_summary(
        efficientnet_path,
        (
            "IMAGE_SIZE",
            "BATCH_SIZE",
            "NUM_EPOCHS",
            "EARLY_STOPPING_PATIENCE",
        ),
    )
    print_notebook_summary(
        densenet_path,
        (
            "IMAGE_SIZE",
            "BATCH_SIZE",
            "PATIENCE",
            "PHASE1_LR",
            "PHASE2_LR",
            "PHASE3_LR",
            "PHASE1_EPOCHS",
            "PHASE2_EPOCHS",
            "PHASE3_EPOCHS",
        ),
    )
    skipped_total = len(quality_skipped) + len(efficient_skipped) + len(dense_skipped)
    if skipped_total:
        print(f"\n  Notebook cells skipped by AST inspection: {skipped_total}")

    print_model_inventory(
        config_values,
        literal_values(inference_path),
    )
    print("\nNo datasets, model weights, or inference settings were modified.")


if __name__ == "__main__":
    main()
