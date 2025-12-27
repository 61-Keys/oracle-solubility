# ORACLE - Protein Solubility Predictor

Predict if your protein will express solubly in E. coli using deep learning.

## Installation
```bash
pip install git+https://github.com/61-Keys/oracle-solubility.git
```

## Usage
```python
from oracle import Oracle

predictor = Oracle()
result = predictor.predict("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK...")

print(result.soluble)
print(result.confidence)
result.visualize()
```

## Command Line
```bash
oracle predict MSKGEELFTGVVPILVELDGDVNGHK...
oracle predict MSKGEELFTGVVPILVELDGDVNGHK... --visualize
oracle info
```

## Performance

- Test Accuracy: 66.8%
- Real-world Accuracy: 85.7%
- Training Data: 60,000 proteins from TargetTrack

## License

MIT License
