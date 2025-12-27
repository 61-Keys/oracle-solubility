"""
🧬 ORACLE - Protein Solubility Predictor

Predict if your protein will express solubly in E. coli.

Usage:
    from oracle import Oracle
    
    predictor = Oracle()
    result = predictor.predict("MSKGEELFTGVVPILVELDGDVNGHK...")
    
    print(result.soluble)        # True/False
    print(result.confidence)     # 0.0-1.0
    result.visualize()           # Beautiful visualization
"""

from .predictor import Oracle, PredictionResult

__version__ = "1.0.0"
__author__ = "Asutosh Rath"
__all__ = ["Oracle", "PredictionResult"]