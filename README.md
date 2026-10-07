# SVHN Digit Sequence Predictor

A convolutional neural network (CNN) that classifies individual digits from Google's
**Street View House Numbers (SVHN)** dataset and chains them into a predicted digit
sequence — the kind that appears on house number plates in Google Street View imagery.

**Dataset source:** Stanford UFLDL — http://ufldl.stanford.edu/housenumbers/

---

## Project layout

```
svhn-dataset/
├── venv/                        # isolated Python environment (pip-bootstrapped)
├── data/
│   ├── train_32x32.mat          # SVHN Format 2 — 73 257 training images
│   ├── test_32x32.mat           # SVHN Format 2 — 26 032 test images
│   ├── norm_mean.npy            # per-channel RGB mean  (saved after first run)
│   └── norm_std.npy             # per-channel RGB std   (saved after first run)
├── models/
│   ├── svhn_model.keras         # final saved model
│   ├── svhn_best.keras          # best checkpoint (highest val_accuracy)
│   ├── training_log.csv         # epoch-by-epoch metrics
│   └── training_results.png    # loss / accuracy / per-digit accuracy plots
├── svhn_classifier.py           # main pipeline  (train + evaluate + demo)
└── README.md
```

---

## Quick start

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Quick training run (5 000 samples, ~15 min on CPU)
python svhn_classifier.py --quick

# 3. Full training run (73 257 samples, expected ~93-95% accuracy)
python svhn_classifier.py
```

All commands can also be run without activating the venv:
```bash
./venv/bin/python svhn_classifier.py --quick
```

---

## Execution steps — `svhn_classifier.py`

The script runs 8 clearly labelled steps in sequence:

| Step | What happens | Key decision |
|------|-------------|--------------|
| **0** | Compatibility check | Verifies Python ≥ 3.8 and all package minimum versions before doing any work; fails fast with a clear error rather than crashing mid-run |
| **1** | Download data | Fetches `train_32x32.mat` and `test_32x32.mat` from Stanford UFLDL; skips download if files are already cached |
| **2** | Load & preprocess | Loads `.mat` files, transposes axes, remaps labels, normalises pixel values |
| **3** | Build model | Constructs the CNN + prints `model.summary()` |
| **4** | Train | Runs `model.fit` with 4 callbacks (see Training section) |
| **5** | Evaluate | Reports overall and per-digit test accuracy |
| **6** | Save model | Writes `models/svhn_model.keras` for later inference |
| **7** | Sequence demo | Assembles digit patches into mock house numbers and runs sequence prediction |
| **8** | Visualise | Saves `models/training_results.png` |

---

## Dataset — SVHN Format 2

**Why Format 2 (not Format 1)?**
SVHN comes in two formats:

- **Format 1** — full street-view photos with variable-length digit sequences and
  per-digit bounding box annotations in `digitStruct.mat`. Needs a detector (YOLO, SSD,
  or sliding window) before classification.
- **Format 2** — pre-cropped 32×32 RGB images, one digit per image. Directly usable
  for training a classifier without a detection stage.

Format 2 was chosen because it lets us focus the model entirely on digit recognition.
Sequence prediction is then achieved by feeding ordered patches (one per digit position)
through the same classifier and concatenating the results. This is practical, easy to
reason about, and achieves state-of-the-art single-digit accuracy without bespoke
detection code.

### Label quirk
The SVHN `.mat` file uses the integer **10** to represent the digit **0** (a MATLAB
1-based indexing artefact). The loader remaps `y[y == 10] = 0` so labels are
consistently in `[0, 9]`.

### Axis quirk
MATLAB stores arrays in column-major (Fortran) order, so the raw array shape is
`(height, width, channels, N)` = `(32, 32, 3, N)`. NumPy and Keras expect
`(N, height, width, channels)`, so we call `.transpose(3, 0, 1, 2)`.

---

## Preprocessing

**Per-channel mean/std normalisation** is applied after loading:

```
X_normalised = (X - mean) / std
```

- `mean` and `std` are computed **only on the training set** (shape `(3,)`, one value
  per RGB channel). Applying training statistics to the test set prevents data leakage.
- Both vectors are saved to `data/norm_mean.npy` / `data/norm_std.npy` so `predict.py`
  can reproduce the exact same normalisation at inference time without access to the
  training data.
- A small epsilon (`1e-7`) is added to `std` to guard against division by zero on
  near-constant channels.

---

## Model architecture

The model is a **VGG-style CNN** — a stack of double convolution blocks followed by
pooling, ending in a Global Average Pooling head rather than Flatten + large Dense layers.

```
Input  (32 × 32 × 3)
│
├── Augmentation (training only)
│     RandomRotation  ±8 %
│     RandomZoom      ±10 %
│     RandomTranslation ±10 % (height & width)
│
├── Block 1  ──────────────────────────────── 32 × 32 → 16 × 16
│     Conv2D  64 filters, 3×3, same padding   (use_bias=False)
│     BatchNormalization
│     ReLU
│     Conv2D  64 filters, 3×3, same padding   (use_bias=False)
│     BatchNormalization
│     ReLU
│     MaxPooling2D  2×2
│     Dropout  0.20
│
├── Block 2  ──────────────────────────────── 16 × 16 → 8 × 8
│     Conv2D  128 filters, 3×3, same padding  (use_bias=False)
│     BatchNormalization
│     ReLU
│     Conv2D  128 filters, 3×3, same padding  (use_bias=False)
│     BatchNormalization
│     ReLU
│     MaxPooling2D  2×2
│     Dropout  0.20
│
├── Block 3  ──────────────────────────────── 8 × 8 → 4 × 4
│     Conv2D  256 filters, 3×3, same padding  (use_bias=False)
│     BatchNormalization
│     ReLU
│     Conv2D  256 filters, 3×3, same padding  (use_bias=False)
│     BatchNormalization
│     ReLU
│     MaxPooling2D  2×2
│     Dropout  0.30
│
├── Head
│     GlobalAveragePooling2D               (4×4×256 → 256)
│     Dense  256, ReLU
│     Dropout  0.40
│     Dense  10,  Softmax                  ← output: P(digit | image)
│
Output  (10,)  — probability over digits 0–9
```

### Layer count

| Category | Layers | Count |
|----------|--------|-------|
| Augmentation | RandomRotation, RandomZoom, RandomTranslation | 3 |
| Convolutional (feature extraction) | 6 × Conv2D + 6 × BatchNorm + 6 × ReLU | 18 |
| Pooling | 3 × MaxPooling2D | 3 |
| Spatial dropout | 3 × Dropout (after each pool) | 3 |
| Head | GlobalAveragePooling2D + Dense(256) + Dropout(0.4) + Dense(10) | 4 |
| **Total trainable layers** | | **~1.21 M trainable params** |

### Why these design choices?

**VGG-style double conv blocks**
Each block applies two 3×3 convolutions before pooling. Two stacked 3×3 filters have
the same receptive field as one 5×5 filter but use fewer parameters and include an extra
non-linearity between them, giving the network more representational power per parameter.

**Filter doubling (64 → 128 → 256)**
Spatial resolution halves at each pooling step (32→16→8→4), so we double the number of
filters to keep the total information capacity roughly constant across the network.

**BatchNormalization after every Conv2D**
Normalises the activations of each layer, which accelerates training convergence and
acts as mild regularisation. We set `use_bias=False` on Conv2D because BatchNorm's
`beta` parameter plays the role of a bias — keeping both would be redundant.

**GlobalAveragePooling2D instead of Flatten**
After Block 3 the spatial maps are 4×4×256. Flattening would give 4,096 units feeding
into a Dense layer, adding ~1 M extra parameters. GAP collapses each feature map to a
single number (its mean), giving a 256-dim vector with zero additional parameters.
This strongly regularises the network and makes it robust to small spatial shifts.

**Dropout schedule (0.20 / 0.20 / 0.30 / 0.40)**
Dropout rate increases as we go deeper. Early convolutional layers learn general
low-level features (edges, colours) that transfer well; aggressive dropout there would
destroy useful shared representations. The final dense layer is the bottleneck where
overfitting is most likely, so we use the highest rate (0.40) there.

**Augmentation inside the model graph**
Keras 3 augmentation layers (`RandomRotation`, `RandomZoom`, `RandomTranslation`) are
applied only during `model.fit` (i.e., `training=True`). At inference time they are
automatically no-ops. This keeps augmentation logic tightly coupled to the model
definition and eliminates the need for a separate data-generator object.

**Adam with lr=3e-4**
The default Adam learning rate (1e-3) can overshoot on small datasets. 3e-4 is the
commonly accepted "safe" default for CNNs on image tasks — large enough to converge
in ~30 epochs, small enough not to oscillate.

---

## Training callbacks

| Callback | Config | Why |
|----------|--------|-----|
| `EarlyStopping` | monitor `val_accuracy`, patience=8, restore best weights | Prevents overfitting without needing to hand-tune the number of epochs; brings back the best checkpoint automatically |
| `ReduceLROnPlateau` | monitor `val_loss`, factor=0.5, patience=4, min=1e-6 | Halves the learning rate whenever validation loss plateaus for 4 epochs; lets the optimizer take finer steps when close to a minimum |
| `ModelCheckpoint` | saves `models/svhn_best.keras` on every val_accuracy improvement | Guarantees we keep the best-ever model even if training diverges in later epochs |
| `CSVLogger` | writes `models/training_log.csv` | Persistent epoch-by-epoch record that can be re-plotted or audited without rerunning training |

---

## Sequence prediction

```
Format-2 training → per-digit classifier
                         │
             ┌───────────┴────────────┐
             │   Sequence prediction   │
             │                         │
  image of   │  [crop digit 1]  ──→  CNN ──→ "4"
  house      │  [crop digit 2]  ──→  CNN ──→ "7"   →  "473"
  number 473 │  [crop digit 3]  ──→  CNN ──→ "3"
             └─────────────────────────┘
```

The `predict_sequence` function in `svhn_classifier.py` (and `predict_batch` in
`predict.py`) accept an **ordered list of 32×32 patches**, normalise them using the
saved training statistics, run a single batched forward pass, and return the
concatenated digit string plus per-digit softmax confidence scores.

**Real-world Format-1 workflow** (extension path):
1. Load the full street-view image
2. Parse `digitStruct.mat` to get bounding boxes for each digit
3. Crop and resize each bounding box region to 32×32
4. Pass the sorted list to `predict_sequence`
5. Read off the house number string

---

## Environment

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12.3 | |
| TensorFlow | 2.21.0 | Backend for Keras 3 |
| Keras | 3.14.0 | Standalone Keras 3 (`import keras`) |
| NumPy | 2.4.4 | |
| SciPy | 1.17.1 | Used only for `scipy.io.loadmat` |
| h5py | 3.14.0 | Required by Keras model serialisation |
| Matplotlib | 3.10.9 | Plot generation |
| scikit-learn | 1.8.0 | Available for future metric extensions |
| Hardware | CPU only | 8 cores, 31 GB RAM, x86_64 |

**Why a virtualenv?**
The system Python (Ubuntu 24.04) has no `pip` and no `sudo` access was available.
The venv was created with `python3 -m venv --without-pip venv` and pip was then
bootstrapped via `get-pip.py`. All project dependencies live inside `venv/` and do
not touch the system Python.

---

## Expected results

| Mode | Train samples | Test samples | Expected accuracy |
|------|--------------|-------------|-------------------|
| `--quick` | 5 000 | 1 000 | ~55–70 % (limited data) |
| Full | 73 257 | 26 032 | ~93–95 % |

Accuracy on the quick mode is intentionally lower — 5 000 samples is roughly 7 % of
the full training set, so the model underfits. Its purpose is fast iteration and
smoke-testing, not peak performance.
