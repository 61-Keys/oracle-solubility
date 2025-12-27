"""
Main Oracle API - Clean and simple like Anthropic's SDK.
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import warnings

from .model import OracleNet
from .features import FeatureExtractor
from .visualize import Visualizer

warnings.filterwarnings("ignore")


@dataclass
class PredictionResult:
    """
    Result of a solubility prediction.
    
    Attributes:
        sequence: Input protein sequence
        soluble: Whether protein is predicted soluble
        confidence: Prediction confidence (0-1)
        solubility_score: Probability of being soluble (0-1)
        features: Computed sequence features
        recommendations: List of recommendations
    """
    sequence: str
    soluble: bool
    confidence: float
    solubility_score: float
    features: Dict[str, float]
    recommendations: List[str]
    
    _visualizer: Optional[Any] = None
    
    def __repr__(self):
        status = "✅ SOLUBLE" if self.soluble else "❌ INSOLUBLE"
        return f"PredictionResult({status}, confidence={self.confidence:.1%})"
    
    def visualize(self, style: str = "full"):
        """
        Visualize the prediction results.
        
        Args:
            style: "full", "minimal", "radar", "hydropathy", or "composition"
        """
        if self._visualizer is None:
            self._visualizer = Visualizer()
        
        self._visualizer.visualize(self, style=style)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "length": len(self.sequence),
            "soluble": self.soluble,
            "confidence": self.confidence,
            "solubility_score": self.solubility_score,
            "features": self.features,
            "recommendations": self.recommendations
        }
    
    def summary(self) -> str:
        """Get a text summary of the prediction."""
        lines = [
            "=" * 50,
            "🧬 ORACLE Prediction Summary",
            "=" * 50,
            f"",
            f"Sequence Length: {len(self.sequence)} amino acids",
            f"",
            f"Prediction: {'✅ SOLUBLE' if self.soluble else '❌ INSOLUBLE'}",
            f"Confidence: {self.confidence:.1%}",
            f"Solubility Score: {self.solubility_score:.1%}",
            f"",
            "Key Features:",
        ]
        
        for name, value in list(self.features.items())[:6]:
            lines.append(f"  • {name}: {value:.3f}")
        
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  {rec}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class Oracle:
    """
    🧬 ORACLE - Protein Solubility Predictor
    
    Predict if your protein will express solubly in E. coli.
    
    Example:
        >>> from oracle import Oracle
        >>> predictor = Oracle()
        >>> result = predictor.predict("MSKGEELFTGVVPILVELDGDVNGHK...")
        >>> print(result.soluble)
        True
        >>> print(result.confidence)
        0.85
        >>> result.visualize()
    """
    
    def __init__(self, device: str = "auto", verbose: bool = True):
        """
        Initialize Oracle predictor.
        
        Args:
            device: "auto", "cpu", "mps", or "cuda"
            verbose: Print loading messages
        """
        self.verbose = verbose
        self._log("🧬 Initializing ORACLE...")
        
        # Set device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self._log(f"   Device: {self.device}")
        
        # Load models
        self._load_models()
        self._log("✅ Ready!")
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _load_models(self):
        """Load ESM and Oracle models."""
        self._log("   Loading ESM-2 model...")
        
        # Load ESM
        self.esm_model, alphabet = torch.hub.load(
            "facebookresearch/esm:main", 
            "esm2_t6_8M_UR50D",
            verbose=False
        )
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()
        
        # Load Oracle model
        self._log("   Loading ORACLE model...")
        self.feature_extractor = FeatureExtractor()
        
        # Try to load from package data or user's trained model
        model_paths = [
            Path(__file__).parent / "data" / "oracle_model.pt",
            Path.home() / "Projects" / "oracle" / "outputs" / "models" / "oracle_v3_best.pt",
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "Oracle model not found. Please train a model first or download weights."
            )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.oracle_model = OracleNet(
            input_dim=checkpoint['config']['input_dim'],
            hidden_dims=checkpoint['config']['hidden_dims'],
            dropout=0.0
        ).to(self.device)
        self.oracle_model.load_state_dict(checkpoint['model_state_dict'])
        self.oracle_model.eval()
        
        # Load scaler
        scaler_paths = [
            Path(__file__).parent / "data" / "scaler.pkl",
            Path.home() / "Projects" / "oracle" / "data" / "processed" / "scaler_v3.pkl",
        ]
        
        import pickle
        for path in scaler_paths:
            if path.exists():
                with open(path, 'rb') as f:
                    self.scaler = pickle.load(f)
                break
        
        self.metrics = checkpoint.get('metrics', {})
    
    def predict(self, sequence: str) -> PredictionResult:
        """
        Predict solubility of a protein sequence.
        
        Args:
            sequence: Protein sequence (single letter amino acids)
        
        Returns:
            PredictionResult with prediction and analysis
        
        Example:
            >>> result = predictor.predict("MSKGEELFTGVVPILVELDGDVNGHK...")
            >>> print(result)
            PredictionResult(✅ SOLUBLE, confidence=85.0%)
        """
        # Clean sequence
        sequence = self._clean_sequence(sequence)
        
        if len(sequence) < 20:
            raise ValueError("Sequence too short (minimum 20 amino acids)")
        
        if len(sequence) > 1000:
            sequence = sequence[:1000]
        
        # Extract features
        features_dict, feature_array = self.feature_extractor.compute_features(sequence)
        
        # Get ESM embedding
        esm_embedding = self._extract_esm_embedding(sequence)
        
        # Combine features
        full_features = np.array(feature_array + list(esm_embedding), dtype=np.float32)
        full_features = full_features.reshape(1, -1)
        full_features = self.scaler.transform(full_features)
        
        # Predict
        with torch.no_grad():
            logits = self.oracle_model(torch.FloatTensor(full_features).to(self.device))
            probs = torch.softmax(logits, dim=1)
        
        solubility_score = probs[0, 1].item()
        soluble = solubility_score > 0.5
        confidence = max(solubility_score, 1 - solubility_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(features_dict, solubility_score)
        
        return PredictionResult(
            sequence=sequence,
            soluble=soluble,
            confidence=confidence,
            solubility_score=solubility_score,
            features=features_dict,
            recommendations=recommendations
        )
    
    def predict_batch(self, sequences: List[str]) -> List[PredictionResult]:
        """
        Predict solubility for multiple sequences.
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            List of PredictionResults
        """
        return [self.predict(seq) for seq in sequences]
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean and validate sequence."""
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = ''.join(c for c in sequence.upper() if c in valid_aa)
        return sequence
    
    def _extract_esm_embedding(self, sequence: str) -> np.ndarray:
        """Extract ESM-2 embedding."""
        seq = sequence[:400]
        batch_data = [("seq", seq)]
        _, _, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[6])
            embedding = results["representations"][6][0, 1:len(seq)+1].mean(dim=0)
        
        return embedding.cpu().numpy()
    
    def _generate_recommendations(self, features: Dict, score: float) -> List[str]:
        """Generate recommendations based on features."""
        recs = []
        
        if score < 0.5:
            recs.append("💡 Consider using solubility tags (MBP, SUMO, or Thioredoxin)")
        
        if features.get('Hydrophobicity', 0) > 0.45:
            recs.append("⚠️ High hydrophobicity - try MBP fusion or lower expression temperature")
        
        if features.get('Aggregation Prone', 0) > 0.45:
            recs.append("⚠️ Aggregation-prone - express at 16-18°C")
        
        if features.get('Cysteine', 0) > 0.03:
            recs.append("🔗 Contains cysteines - consider Origami/SHuffle strains for disulfide bonds")
        
        if features.get('Length', 0) > 500:
            recs.append("📏 Large protein - consider domain truncations")
        
        if features.get('Disorder Prone', 0) > 0.4:
            recs.append("🌀 Disordered regions - may need binding partner for stability")
        
        if not recs:
            recs.append("✅ No major concerns - good candidate for standard expression")
        
        return recs
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "version": "3.0",
            "training_data": "TargetTrack (PSI:Biology)",
            "training_samples": 60000,
            "test_accuracy": self.metrics.get('test_accuracy', 'N/A'),
            "test_f1": self.metrics.get('test_f1', 'N/A'),
            "test_auc": self.metrics.get('test_auc', 'N/A'),
        }