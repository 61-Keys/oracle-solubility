<div align="center">

# 🧬 ORACLE

### Optimized Recombinant Assessment for Cellular Laboratory Expression

**Predict protein solubility in E. coli using deep learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/61-Keys/oracle-solubility?style=social)](https://github.com/61-Keys/oracle-solubility)

[Installation](#installation) • [Usage](#usage) • [Performance](#performance) • [How It Works](#how-it-works) • [Citation](#citation)

---

</div>

##  What is ORACLE?

ORACLE is a deep learning tool that predicts whether your protein will express as **soluble** or form **inclusion bodies** in E. coli. It helps researchers prioritize which proteins to express experimentally, saving time and resources.

### Why use ORACLE?

- **Save weeks of lab work** - Know beforehand which proteins are likely to fail
- **State-of-the-art** - Uses ESM-2 protein language model embeddings
- **Beautiful visualizations** - Understand why your protein may or may not express
- **Actionable recommendations** - Get suggestions to improve expression

---

## 📦 Installation
```bash
pip install git+https://github.com/61-Keys/oracle-solubility.git
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- ~500MB disk space (for ESM-2 model)

---

## Usage

### Python API
```python
from oracle import Oracle

# Initialize predictor (downloads ESM-2 model on first run)
predictor = Oracle()

# Predict solubility
sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK..."
result = predictor.predict(sequence)

# Access results
print(result.soluble)          # True or False
print(result.confidence)       # 0.0 to 1.0
print(result.solubility_score) # Probability of being soluble
print(result.recommendations)  # List of suggestions

# Get detailed summary
print(result.summary())

# Visualize results
result.visualize()          # Full dashboard
result.visualize("minimal") # Compact card view
```

### Command Line Interface
```bash
# Basic prediction
oracle predict MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK...

# With visualization
oracle predict MSKGEELFTGVVPILVELDGDVNGHK... --visualize

# Minimal visualization
oracle predict MSKGEELFTGVVPILVELDGDVNGHK... --visualize --minimal

# Model information
oracle info
```

---

## 📊 Performance

### Benchmark Results

| Metric | Value |
|--------|-------|
| **Test Set Accuracy** | 66.8% |
| **F1 Score** | 0.67 |
| **AUC-ROC** | 0.73 |

### Real-World Protein Tests

| Protein | Expected | Predicted | Correct |
|---------|----------|-----------|---------|
| GFP | Soluble | ✅ Soluble (60%) | ✅ |
| Ubiquitin | Soluble | ✅ Soluble (75%) | ✅ |
| Thioredoxin | Soluble | ✅ Soluble (86%) | ✅ |
| SUMO | Soluble | ✅ Soluble (88%) | ✅ |
| MBP | Soluble | ✅ Soluble (71%) | ✅ |
| GPCR (membrane) | Insoluble | ✅ Insoluble (74%) | ✅ |
| Amyloid Beta | Insoluble | ✅ Insoluble (82%) | ✅ |
| Alpha-Synuclein | Insoluble | ✅ Insoluble (78%) | ✅ |
| Spider Silk | Insoluble | ✅ Insoluble (91%) | ✅ |
| Prion Protein | Insoluble | ✅ Insoluble (85%) | ✅ |

**Overall: 12/14 correct**

---

## How It Works

ORACLE combines two types of features:

### 1. ESM-2 Embeddings (320 features)
- Uses Facebook's ESM-2 protein language model
- Captures evolutionary and structural information
- Pre-trained on millions of protein sequences

### 2. Biophysical Features (14 features)
- Sequence length
- Hydrophobicity
- Charge distribution
- Disorder propensity
- Aggregation propensity
- Amino acid composition

### Architecture
```
Input Sequence
      ↓
┌─────────────────┐    ┌──────────────────┐
│   ESM-2 Model   │    │ Sequence Features │
│  (320-dim emb)  │    │    (14 features)  │
└────────┬────────┘    └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    ↓
            ┌───────────────┐
            │  Neural Net   │
            │ 512→256→128→2 │
            └───────┬───────┘
                    ↓
         Soluble / Insoluble
```

---

##  Training Data

ORACLE was trained on **60,000 proteins** from [TargetTrack (PSI:Biology)](https://zenodo.org/records/821654):

-  **35 research centers** worldwide
-  **Real experimental outcomes** (not computational predictions)
-  **Balanced dataset** (30,000 soluble, 30,000 insoluble)
-  **17 years** of structural genomics data

### Label Definitions

| Status | Label | Meaning |
|--------|-------|---------|
| In PDB | Soluble | Successfully crystallized |
| Purified | Soluble | Expressed and purified |
| Soluble | Soluble | Confirmed soluble expression |
| Work Stopped | Insoluble | Failed to express/purify |
| Cloned | Insoluble | Never made it past cloning |

---

## 🔧 API Reference

### `Oracle(device="auto", verbose=True)`

Initialize the predictor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | "auto" | Device to use: "auto", "cpu", "mps", "cuda" |
| `verbose` | bool | True | Print loading messages |

### `Oracle.predict(sequence) -> PredictionResult`

Predict solubility for a protein sequence.

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence` | str | Protein sequence (single letter amino acids) |

### `PredictionResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `.soluble` | bool | True if predicted soluble |
| `.confidence` | float | Prediction confidence (0-1) |
| `.solubility_score` | float | Probability of being soluble |
| `.features` | dict | Computed sequence features |
| `.recommendations` | list | Suggestions for improving expression |
| `.visualize(style)` | method | Show visualization ("full" or "minimal") |
| `.summary()` | method | Get text summary |
| `.to_dict()` | method | Convert to dictionary |

---

## 💡 Recommendations

ORACLE provides actionable recommendations based on sequence analysis:

| Issue | Recommendation |
|-------|----------------|
| High hydrophobicity | Try MBP or SUMO fusion tags |
| Aggregation-prone | Express at 16-18°C |
| Contains cysteines | Use Origami/SHuffle strains |
| Large protein (>500 aa) | Consider domain truncations |
| Disordered regions | May need binding partner |

---

## 📖 Citation

If you use ORACLE in your research, please cite:
```bibtex
@software{oracle2024,
  title={ORACLE: Protein Solubility Predictor},
  author={Rath, Asutosh},
  year={2024},
  url={https://github.com/61-Keys/oracle-solubility}
}
```

### Data Source
```bibtex
@article{targettrack,
  title={TargetTrack: A resource for tracking targets in structural genomics},
  journal={Zenodo},
  doi={10.5281/zenodo.821654}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.



<div align="center">

**Made with 🧬 by [Asutosh Rath](https://github.com/61-Keys)**

</div>
