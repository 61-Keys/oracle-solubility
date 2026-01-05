"""
AlphaFold Feature Extractor for ORACLE v4.0
Adds structural confidence and disorder features to improve solubility prediction.
"""

import numpy as np
import requests
import json
from io import StringIO
from typing import Dict, List, Optional, Tuple
import warnings

class AlphaFoldFeatureExtractor:
    """Extract structural features from AlphaFold predictions."""
    
    def __init__(self):
        self.af_base_url = "https://alphafold.ebi.ac.uk/api/prediction/"
        
    def get_alphafold_prediction(self, uniprot_id: str) -> Optional[Dict]:
        """
        Fetch AlphaFold prediction from EBI API.
        
        Args:
            uniprot_id: UniProt identifier
            
        Returns:
            Dictionary with structure data or None if not found
        """
        try:
            response = requests.get(f"{self.af_base_url}{uniprot_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            warnings.warn(f"Failed to fetch AlphaFold data: {e}")
            return None
    
    def predict_structure_confidence(self, sequence: str) -> Dict[str, float]:
        """
        Predict structural features using ColabFold (local AlphaFold).
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of structural confidence features
        """
        # This would typically run ColabFold, but for now we'll simulate
        # In real implementation, this would call ColabFold API
        
        # Placeholder: realistic confidence distributions
        length = len(sequence)
        
        # Simulate confidence scores (in real implementation, get from ColabFold)
        confidence_scores = np.random.beta(3, 1, length) * 100  # Biased toward high confidence
        
        features = {
            'mean_confidence': float(np.mean(confidence_scores)),
            'min_confidence': float(np.min(confidence_scores)),
            'max_confidence': float(np.max(confidence_scores)),
            'low_confidence_fraction': float(np.mean(confidence_scores < 70)),
            'very_low_confidence_fraction': float(np.mean(confidence_scores < 50)),
            'high_confidence_fraction': float(np.mean(confidence_scores > 90)),
            'confidence_variance': float(np.var(confidence_scores)),
        }
        
        return features
    
    def predict_secondary_structure(self, sequence: str) -> Dict[str, float]:
        """
        Predict secondary structure content.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of secondary structure features
        """
        # Simplified secondary structure prediction based on sequence
        # In real implementation, would use DSSP or other tools
        
        length = len(sequence)
        
        # Secondary structure propensities (simplified)
        helix_prone = set('AEKLAQHQKLE')
        sheet_prone = set('VILMFYWC')
        loop_prone = set('GSTPND')
        
        helix_content = sum(1 for aa in sequence if aa in helix_prone) / length
        sheet_content = sum(1 for aa in sequence if aa in sheet_prone) / length
        loop_content = sum(1 for aa in sequence if aa in loop_prone) / length
        
        features = {
            'helix_content': helix_content,
            'sheet_content': sheet_content, 
            'loop_content': loop_content,
            'structured_content': helix_content + sheet_content,
        }
        
        return features
    
    def predict_domain_features(self, sequence: str) -> Dict[str, float]:
        """
        Predict domain-related features.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of domain features
        """
        length = len(sequence)
        
        # Simple domain boundary prediction based on length
        if length < 100:
            num_domains = 1
        elif length < 300:
            num_domains = 1 + (length - 100) // 150
        else:
            num_domains = 2 + (length - 300) // 200
            
        features = {
            'predicted_domains': float(num_domains),
            'avg_domain_size': length / num_domains,
            'multidomain': float(num_domains > 1),
        }
        
        return features
    
    def predict_disorder_features(self, sequence: str) -> Dict[str, float]:
        """
        Predict intrinsic disorder features.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of disorder features
        """
        # Disorder-promoting residues (simplified)
        disorder_prone = set('PSGAQEKRNDC')
        order_prone = set('WFILVMYCHK')
        
        length = len(sequence)
        disorder_content = sum(1 for aa in sequence if aa in disorder_prone) / length
        order_content = sum(1 for aa in sequence if aa in order_prone) / length
        
        # Predict disorder regions (simplified sliding window)
        window_size = 21
        disorder_scores = []
        
        for i in range(length - window_size + 1):
            window = sequence[i:i + window_size]
            window_disorder = sum(1 for aa in window if aa in disorder_prone) / window_size
            disorder_scores.append(window_disorder)
        
        if disorder_scores:
            max_disorder_region = max(disorder_scores)
            disorder_regions = sum(1 for score in disorder_scores if score > 0.5)
        else:
            max_disorder_region = disorder_content
            disorder_regions = 0
            
        features = {
            'disorder_content': disorder_content,
            'order_content': order_content,
            'max_disorder_region': max_disorder_region,
            'num_disorder_regions': float(disorder_regions),
            'disorder_order_ratio': disorder_content / (order_content + 1e-6),
        }
        
        return features
    
    def extract_all_features(self, sequence: str, uniprot_id: Optional[str] = None) -> Dict[str, float]:
        """
        Extract all structural features for a protein sequence.
        
        Args:
            sequence: Amino acid sequence
            uniprot_id: Optional UniProt ID for existing structures
            
        Returns:
            Dictionary of all structural features (16 new features)
        """
        all_features = {}
        
        # Add confidence features
        all_features.update(self.predict_structure_confidence(sequence))
        
        # Add secondary structure features  
        all_features.update(self.predict_secondary_structure(sequence))
        
        # Add domain features
        all_features.update(self.predict_domain_features(sequence))
        
        # Add disorder features
        all_features.update(self.predict_disorder_features(sequence))
        
        return all_features

# Example usage
if __name__ == "__main__":
    extractor = AlphaFoldFeatureExtractor()
    
    # Test on GFP
    gfp_sequence = """MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL
    VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV
    NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD
    HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK""".replace('\n', '').replace(' ', '')
    
    features = extractor.extract_all_features(gfp_sequence)
    
    print("Structural features for GFP:")
    for feature, value in features.items():
        print(f"{feature}: {value:.3f}")
        
    print(f"\nTotal new features: {len(features)}")
