"""Feature extraction for protein sequences."""

import numpy as np
from typing import Dict, List, Tuple


class FeatureExtractor:
    """Extract biophysical features from protein sequences."""
    
    # Amino acid groups
    HYDROPHOBIC = set("AILMFVPWG")
    CHARGED = set("DEKR")
    POSITIVE = set("KRH")
    NEGATIVE = set("DE")
    POLAR = set("STNQYC")
    AROMATIC = set("FWY")
    DISORDER_PRONE = set("PSGAQEK")
    AGGREGATION_PRONE = set("VILMFYW")
    
    # Kyte-Doolittle hydropathy scale
    KD_SCALE = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    def compute_features(self, sequence: str) -> Tuple[Dict[str, float], List[float]]:
        """
        Compute all features for a sequence.
        
        Returns:
            Tuple of (feature_dict, feature_array)
        """
        seq = sequence.upper()
        length = len(seq)
        
        if length == 0:
            return {}, [0.0] * 14
        
        features = {
            'Length': length,
            'Hydrophobicity': self._fraction(seq, self.HYDROPHOBIC),
            'Charged': self._fraction(seq, self.CHARGED),
            'Positive': self._fraction(seq, self.POSITIVE),
            'Negative': self._fraction(seq, self.NEGATIVE),
            'Polar': self._fraction(seq, self.POLAR),
            'Aromatic': self._fraction(seq, self.AROMATIC),
            'Disorder Prone': self._fraction(seq, self.DISORDER_PRONE),
            'Aggregation Prone': self._fraction(seq, self.AGGREGATION_PRONE),
            'Cysteine': seq.count('C') / length,
            'Proline': seq.count('P') / length,
            'Glycine': seq.count('G') / length,
            'Net Charge': (self._count(seq, self.POSITIVE) - self._count(seq, self.NEGATIVE)) / length,
            'Entropy': self._sequence_entropy(seq),
        }
        
        # Feature array for model (must match training order)
        feature_array = [
            features['Length'],
            features['Hydrophobicity'],
            features['Charged'],
            features['Positive'],
            features['Negative'],
            features['Polar'],
            features['Aromatic'],
            features['Disorder Prone'],
            features['Aggregation Prone'],
            features['Cysteine'],
            features['Proline'],
            features['Glycine'],
            features['Net Charge'],
            features['Entropy'],
        ]
        
        return features, feature_array
    
    def compute_hydropathy_profile(self, sequence: str, window: int = 9) -> Tuple[List[int], List[float]]:
        """Compute hydropathy profile along sequence."""
        seq = sequence.upper()
        
        if len(seq) < window:
            return [], []
        
        positions = []
        scores = []
        
        for i in range(len(seq) - window + 1):
            win = seq[i:i + window]
            score = np.mean([self.KD_SCALE.get(aa, 0) for aa in win])
            positions.append(i + window // 2)
            scores.append(score)
        
        return positions, scores
    
    def get_aa_composition(self, sequence: str) -> Dict[str, float]:
        """Get amino acid composition as percentages."""
        seq = sequence.upper()
        aa_list = "ACDEFGHIKLMNPQRSTVWY"
        return {aa: seq.count(aa) / len(seq) * 100 for aa in aa_list}
    
    def _fraction(self, seq: str, aa_set: set) -> float:
        return sum(1 for aa in seq if aa in aa_set) / len(seq)
    
    def _count(self, seq: str, aa_set: set) -> int:
        return sum(1 for aa in seq if aa in aa_set)
    
    def _sequence_entropy(self, seq: str) -> float:
        """Calculate Shannon entropy of sequence."""
        aa_counts = [seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"]
        length = len(seq)
        probs = [c / length for c in aa_counts if c > 0]
        return -sum(p * np.log2(p) for p in probs) if probs else 0