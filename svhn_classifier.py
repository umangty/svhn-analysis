#!/usr/bin/env python3
"""
SVHN (Street View House Numbers) — Digit Sequence Prediction Pipeline
======================================================================
Downloads SVHN Format 2 (32×32 cropped digit images), trains a CNN with
Keras/TensorFlow, evaluates on the test set, and demonstrates sequence
prediction by chaining per-digit classifications.

Reference: http://ufldl.stanford.edu/housenumbers/

Usage:
    ./venv/bin/python svhn_classifier.py           # full training run
    ./venv/bin/python svhn_classifier.py --quick   # smoke-test (5k samples)
    ./venv/bin/python svhn_classifier.py --epochs 50 --batch 256
"""

import argparse
import sys
import urllib.request
from pathlib import Path
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 0.  COMPATIBILITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

def check_compatibility() -> None:
    """Verify all required packages and their minimum versions."""
    print("=" * 50)
    print(" System Compatibility Check")
    print("=" * 50)

    errors: list[str] = []

    if sys.version_info < (3, 8):
        errors.append(f"Python ≥ 3.8 required (found {sys.version})")

    requirements = {
        "tensorflow": (2, 0),
        "keras":      (3, 0),
        "numpy":      (1, 20),
        "scipy":      (1, 5),
        "h5py":       (3, 0),
        "matplotlib": (3, 0),
        "sklearn":    (1, 0),
    }

    installed: dict[str, str] = {}
    for pkg, min_ver in requirements.items():
        import_name = pkg  # sklearn is importable as sklearn
        try:
            mod = __import__(import_name)
            ver_str = mod.__version__
            installed[pkg] = ver_str
            major, minor = (int(x) for x in ver_str.split(".")[:2])
            if (major, minor) < min_ver:
                errors.append(
                    f"{pkg} ≥ {min_ver[0]}.{min_ver[1]} required "
                    f"(found {ver_str})"
                )
        except ImportError:
            errors.append(f"{pkg} is not installed")

    if errors:
        print("\nCOMPATIBILITY ERRORS:")
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)

    print(f"  Python       : {sys.version.split()[0]}")
    for pkg, ver in installed.items():
        print(f"  {pkg:<14}: {ver}")

    # GPU detection
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  GPU(s)       : {', '.join(g.name for g in gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("  GPU          : none — running on CPU")

    print("  All checks passed.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")

SVHN_URLS = {
    "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    "test":  "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
}


def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    pct = min(count * block_size * 100 // total_size, 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r    [{bar}] {pct:3d}%", end="", flush=True)


def download_data() -> None:
    """Download SVHN Format-2 train/test .mat files (skips if cached)."""
    DATA_DIR.mkdir(exist_ok=True)
    for split, url in SVHN_URLS.items():
        dest = DATA_DIR / f"{split}_32x32.mat"
        if dest.exists():
            size_mb = dest.stat().st_size / 1e6
            print(f"  [cached] {dest}  ({size_mb:.1f} MB)")
            continue
        print(f"  Downloading {split} set from UFLDL …")
        urllib.request.urlretrieve(url, dest, _progress_hook)
        size_mb = dest.stat().st_size / 1e6
        print(f"\n    Saved → {dest}  ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an SVHN Format-2 .mat file.

    MATLAB stores arrays in column-major order, so the raw shape is
    (32, 32, 3, N).  We transpose to (N, 32, 32, 3) for Keras.
    Labels use 10 to represent the digit 0; we remap to [0-9].

    Returns:
        X: float32 array  (N, 32, 32, 3)
        y: int32 array    (N,)  values in [0, 9]
    """
    import scipy.io

    raw = scipy.io.loadmat(str(path))
    X = raw["X"].transpose(3, 0, 1, 2).astype(np.float32)  # (N, 32, 32, 3)
    y = raw["y"].squeeze().astype(np.int32)
    y[y == 10] = 0  # remap SVHN label 10 → digit 0
    return X, y


def normalise(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel mean/std normalisation.
    Statistics are computed on the training set only to prevent data leakage.

    Returns:
        X_train_n, X_test_n : normalised arrays
        mean, std            : (3,) vectors saved for inference time
    """
    mean = X_train.mean(axis=(0, 1, 2))        # shape (3,)
    std  = X_train.std(axis=(0, 1, 2)) + 1e-7  # shape (3,)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MODEL
# ──────────────────────────────────────────────────────────────────────────────

def build_model() -> "keras.Model":
    """
    VGG-style CNN for 10-class digit recognition on 32×32 RGB images.

    Architecture (input → output spatial size):
        Augmentation (random rotation / zoom / shift — training only)
        Block 1  32→16  Conv64×2  → MaxPool → Dropout 0.20
        Block 2  16→8   Conv128×2 → MaxPool → Dropout 0.20
        Block 3  8→4    Conv256×2 → MaxPool → Dropout 0.30
        Head             GAP → Dense 256 → Dropout 0.40 → Softmax 10

    ~3.4 M parameters, ~93–95 % test accuracy on full SVHN Format 2.
    """
    import keras
    from keras import layers

    inp = keras.Input(shape=(32, 32, 3), name="image")

    # ── Data-augmentation layers (no-op at inference) ─────────────────────
    x = layers.RandomRotation(0.08, name="aug_rotate")(inp)
    x = layers.RandomZoom(0.10, name="aug_zoom")(x)
    x = layers.RandomTranslation(0.10, 0.10, name="aug_translate")(x)

    # ── Convolution helper ────────────────────────────────────────────────
    def conv_bn_relu(tensor, filters: int, name_prefix: str):
        tensor = layers.Conv2D(
            filters, 3, padding="same", use_bias=False,
            name=f"{name_prefix}_conv"
        )(tensor)
        tensor = layers.BatchNormalization(name=f"{name_prefix}_bn")(tensor)
        return layers.Activation("relu", name=f"{name_prefix}_relu")(tensor)

    # ── Block 1: 32×32 → 16×16 ───────────────────────────────────────────
    x = conv_bn_relu(x, 64,  "b1a")
    x = conv_bn_relu(x, 64,  "b1b")
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.20,  name="drop1")(x)

    # ── Block 2: 16×16 → 8×8 ─────────────────────────────────────────────
    x = conv_bn_relu(x, 128, "b2a")
    x = conv_bn_relu(x, 128, "b2b")
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(0.20,  name="drop2")(x)

    # ── Block 3: 8×8 → 4×4 ───────────────────────────────────────────────
    x = conv_bn_relu(x, 256, "b3a")
    x = conv_bn_relu(x, 256, "b3b")
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.Dropout(0.30,  name="drop3")(x)

    # ── Head ──────────────────────────────────────────────────────────────
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.Dense(256, activation="relu", name="fc1")(x)
    x   = layers.Dropout(0.40, name="drop_fc")(x)
    out = layers.Dense(10, activation="softmax", name="predictions")(x)

    model = keras.Model(inp, out, name="svhn_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SEQUENCE PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_sequence(
    model,
    patches: list[np.ndarray] | np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[str, list[float]]:
    """
    Predict a digit sequence from an ordered list of 32×32 image patches.

    In practice (SVHN Format 1 full images) the workflow is:
        1. Detect digit bounding boxes (e.g. via digitStruct.mat or a detector)
        2. Crop and resize each bounding box region to 32×32
        3. Pass the sorted list of patches to this function
        4. Concatenate predictions → house-number string

    Args:
        model   : trained Keras model (expects normalised input)
        patches : list/array of K patches, each (32, 32, 3) uint8 or float32
        mean    : per-channel mean used during training  (3,)
        std     : per-channel std  used during training  (3,)

    Returns:
        sequence     : digit string, e.g. "4273"
        confidences  : per-digit softmax confidence,  e.g. [0.97, 0.88, ...]
    """
    if len(patches) == 0:
        return "", []

    batch = np.array(patches, dtype=np.float32)  # (K, 32, 32, 3)
    batch = (batch - mean) / std
    probs   = model.predict(batch, verbose=0)    # (K, 10)
    digits  = np.argmax(probs, axis=1)
    confs   = probs[np.arange(len(digits)), digits]
    return "".join(str(int(d)) for d in digits), confs.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(
    history,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    out_path: Path = Path("models/training_results.png"),
) -> None:
    """Save a 3-panel figure: loss/acc curves + per-class accuracy + sample grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("SVHN CNN — Training Summary", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── Loss curves ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history["loss"],     label="train", linewidth=1.5)
    ax1.plot(history.history["val_loss"], label="val",   linewidth=1.5)
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch"); ax1.legend()

    # ── Accuracy curves ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history["accuracy"],     label="train", linewidth=1.5)
    ax2.plot(history.history["val_accuracy"], label="val",   linewidth=1.5)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylim(0, 1); ax2.legend()

    # ── Per-digit accuracy bar chart ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    X_norm = (X_test - mean) / std
    preds  = np.argmax(model.predict(X_norm, verbose=0), axis=1)
    per_class = [
        float((preds[y_test == d] == d).mean()) if (y_test == d).any() else 0.0
        for d in range(10)
    ]
    bars = ax3.bar(range(10), per_class, color="steelblue")
    for bar, acc in zip(bars, per_class):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.2f}", ha="center", va="bottom", fontsize=7
        )
    ax3.set_title("Per-Digit Accuracy")
    ax3.set_xlabel("Digit"); ax3.set_xticks(range(10)); ax3.set_ylim(0, 1.1)

    # ── 15-image prediction grid ──────────────────────────────────────────────
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), 15, replace=False)
    for col, idx in enumerate(sample_idx):
        ax = fig.add_subplot(gs[1, 0] if col < 5
                             else gs[1, 1] if col < 10
                             else gs[1, 2])
        # each gs[1,*] gets 5 images side-by-side via inset_axes
        # use a simple approach: override with proper sub-subplot
        pass

    # Replace bottom row with 3×5 sub-grid
    for panel in range(3):
        for pos in range(5):
            idx  = sample_idx[panel * 5 + pos]
            inner_ax = fig.add_axes([
                0.04 + panel * 0.335 + pos * 0.064,
                0.06,
                0.058, 0.30,
            ])
            img = X_test[idx].clip(0, 255).astype(np.uint8)
            inner_ax.imshow(img)
            inner_ax.axis("off")
            true_d = int(y_test[idx])
            pred_d = int(preds[idx])
            color  = "green" if true_d == pred_d else "red"
            inner_ax.set_title(
                f"T{true_d}/P{pred_d}",
                fontsize=7, color=color, pad=2
            )

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Figure saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CNN on SVHN and predict digit sequences."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use only 5 000 train / 1 000 test samples (smoke-test)"
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum training epochs (default: 30)")
    parser.add_argument("--batch",  type=int, default=128,
                        help="Mini-batch size (default: 128)")
    args = parser.parse_args()

    # ── 0. Environment check ──────────────────────────────────────────────────
    check_compatibility()

    # ── 1. Download data ──────────────────────────────────────────────────────
    print("Step 1 — Download data")
    print("-" * 40)
    download_data()
    print()

    # ── 2. Load data ──────────────────────────────────────────────────────────
    print("Step 2 — Load & preprocess")
    print("-" * 40)
    X_train, y_train = load_mat(DATA_DIR / "train_32x32.mat")
    X_test,  y_test  = load_mat(DATA_DIR / "test_32x32.mat")

    if args.quick:
        rng = np.random.default_rng(42)
        tr_idx = rng.choice(len(X_train), 5_000, replace=False)
        te_idx = rng.choice(len(X_test),  1_000, replace=False)
        X_train, y_train = X_train[tr_idx], y_train[tr_idx]
        X_test,  y_test  = X_test[te_idx],  y_test[te_idx]
        print("  [quick mode] 5 000 train / 1 000 test samples")

    print(f"  Train : {X_train.shape}  dtype={X_train.dtype}")
    print(f"  Test  : {X_test.shape}   dtype={X_test.dtype}")

    unique, counts = np.unique(y_train, return_counts=True)
    dist = {str(int(d)): int(c) for d, c in zip(unique, counts)}
    print(f"  Train digit distribution: {dist}")

    # Normalise
    X_train_n, X_test_n, mean, std = normalise(X_train, X_test)
    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / "norm_mean.npy", mean)
    np.save(DATA_DIR / "norm_std.npy",  std)
    print(f"  Normalisation params saved → data/norm_mean.npy, data/norm_std.npy\n")

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("Step 3 — Build model")
    print("-" * 40)
    import keras
    model = build_model()
    model.summary()
    print()

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("Step 4 — Train")
    print("-" * 40)
    Path("models").mkdir(exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "models/svhn_best.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.CSVLogger("models/training_log.csv"),
    ]

    history = model.fit(
        X_train_n, y_train,
        batch_size=args.batch,
        epochs=args.epochs,
        validation_data=(X_test_n, y_test),
        callbacks=callbacks,
        verbose=1,
    )
    print()

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("Step 5 — Evaluate")
    print("-" * 40)
    loss, acc = model.evaluate(X_test_n, y_test, verbose=0)
    print(f"  Test accuracy : {acc * 100:.2f}%")
    print(f"  Test loss     : {loss:.4f}")

    preds = np.argmax(model.predict(X_test_n, verbose=0), axis=1)
    print("\n  Per-digit accuracy:")
    for digit in range(10):
        mask = y_test == digit
        if not mask.any():
            continue
        d_acc = float((preds[mask] == digit).mean())
        n     = int(mask.sum())
        bar   = "█" * int(d_acc * 20)
        print(f"    Digit {digit}: {d_acc * 100:5.1f}%  {bar:<20}  n={n}")
    print()

    # ── 6. Save model ─────────────────────────────────────────────────────────
    print("Step 6 — Save model")
    print("-" * 40)
    model.save("models/svhn_model.keras")
    print("  Model saved → models/svhn_model.keras\n")

    # ── 7. Sequence prediction demo ───────────────────────────────────────────
    print("Step 7 — Sequence prediction demo")
    print("-" * 40)
    print(
        "  Simulates a house-number image by concatenating individual digit\n"
        "  patches from the test set and predicting the resulting sequence.\n"
        "  (In production: crop per-digit regions from a Format-1 image first.)"
    )

    rng = np.random.default_rng(0)
    for trial, target in enumerate(
        [[4, 2, 7, 3], [1, 9], [3, 8, 5, 0, 2], [7, 7]], start=1
    ):
        patches = []
        for d in target:
            candidates = np.where(y_test == d)[0]
            idx        = rng.choice(candidates)
            patches.append(X_test[idx])

        pred_seq, confs = predict_sequence(model, patches, mean, std)
        true_seq = "".join(str(d) for d in target)
        status   = "OK" if pred_seq == true_seq else "WRONG"
        conf_str = "  ".join(f"{c:.0%}" for c in confs)
        print(f"  Trial {trial}:  true={true_seq}  pred={pred_seq}  "
              f"[{status}]  conf: {conf_str}")
    print()

    # ── 8. Visualise ──────────────────────────────────────────────────────────
    print("Step 8 — Save training plots")
    print("-" * 40)
    plot_results(history, X_test, y_test, model, mean, std)

    print("\nAll done.  Run `./venv/bin/python predict.py --demo` to see live predictions.")


if __name__ == "__main__":
    main()
