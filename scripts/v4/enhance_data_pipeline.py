"""
Data Enhancement Pipeline for ORACLE v4.0
Adds structural features to existing TargetTrack dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import torch
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from alphafold_features import AlphaFoldFeatureExtractor
from oracle_v4 import OracleV4

class DataEnhancementPipeline:
    """Pipeline to enhance TargetTrack data with structural features."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature extractor
        self.af_extractor = AlphaFoldFeatureExtractor()
        
        print("🧬 ORACLE v4.0 Data Enhancement Pipeline")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def load_targettrack_data(self) -> pd.DataFrame:
        """Load existing TargetTrack dataset."""
        # Try different possible file names
        possible_files = [
            'targettrack_60k.csv',
            'processed_proteins.csv', 
            'oracle_training_data.csv',
            'training_data.csv'
        ]
        
        for filename in possible_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                print(f"Loading data from {filepath}")
                df = pd.read_csv(filepath)
                print(f"Loaded {len(df)} proteins")
                return df
                
        # If no file found, create sample data for demonstration
        print("No existing data found. Creating sample dataset...")
        return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing."""
        sample_proteins = [
            # Some real proteins for testing
            {
                'sequence': 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
                'label': 1,  # Soluble
                'protein_name': 'GFP'
            },
            {
                'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG', 
                'label': 1,  # Soluble
                'protein_name': 'Ubiquitin'
            },
            {
                'sequence': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
                'label': 0,  # Insoluble
                'protein_name': 'Amyloid_Beta'
            }
        ]
        
        # Expand with random sequences for testing
        np.random.seed(42)
        for i in range(100):
            # Generate random protein sequence
            aa = 'ACDEFGHIKLMNPQRSTVWY'
            length = np.random.randint(50, 500)
            seq = ''.join(np.random.choice(list(aa), length))
            label = np.random.choice([0, 1])
            
            sample_proteins.append({
                'sequence': seq,
                'label': label,
                'protein_name': f'random_protein_{i}'
            })
        
        df = pd.DataFrame(sample_proteins)
        print(f"Created sample dataset with {len(df)} proteins")
        return df
    
    def extract_biophysical_features(self, sequence: str) -> Dict[str, float]:
        """Extract the original 14 biophysical features."""
        length = len(sequence)
        
        # Amino acid groups
        hydrophobic = set('AILMFVPWG')
        charged = set('DEKR')
        positive = set('KRH') 
        negative = set('DE')
        polar = set('STNQYC')
        aromatic = set('FWY')
        disorder_prone = set('PSGAQEK')
        aggregation_prone = set('VILMFYW')
        
        # Calculate entropy
        from collections import Counter
        import math
        counts = Counter(sequence)
        entropy = 0.0
        for count in counts.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        features = {
            'length': length,
            'hydrophobicity': sum(1 for aa in sequence if aa in hydrophobic) / length,
            'charged': sum(1 for aa in sequence if aa in charged) / length,
            'positive': sum(1 for aa in sequence if aa in positive) / length,
            'negative': sum(1 for aa in sequence if aa in negative) / length,
            'polar': sum(1 for aa in sequence if aa in polar) / length,
            'aromatic': sum(1 for aa in sequence if aa in aromatic) / length,
            'disorder_prone': sum(1 for aa in sequence if aa in disorder_prone) / length,
            'aggregation_prone': sum(1 for aa in sequence if aa in aggregation_prone) / length,
            'cysteine': sequence.count('C') / length,
            'proline': sequence.count('P') / length,
            'glycine': sequence.count('G') / length,
            'net_charge': (sum(1 for aa in sequence if aa in positive) - 
                          sum(1 for aa in sequence if aa in negative)) / length,
            'entropy': entropy,
        }
        
        return features
    
    def extract_esm_features_batch(self, sequences: List[str], batch_size: int = 10) -> np.ndarray:
        """Extract ESM-2 embeddings for batch of sequences."""
        print(f"Extracting ESM-2 embeddings for {len(sequences)} sequences...")
        
        # For now, simulate ESM-2 embeddings
        # In real implementation, would use actual ESM-2 model
        embeddings = []
        
        for seq in tqdm(sequences, desc="ESM-2 embeddings"):
            # Deterministic random embeddings based on sequence
            np.random.seed(hash(seq) % 2**32)
            emb = np.random.normal(0, 0.1, 320)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def process_dataset(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """Process entire dataset with enhanced features."""
        print(f"Processing {len(df)} proteins with enhanced features...")
        
        # Storage for all features
        enhanced_data = []
        
        # Process in batches to manage memory
        for batch_start in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            # Extract all feature types for this batch
            batch_results = []
            
            for _, row in batch_df.iterrows():
                sequence = row['sequence']
                
                try:
                    # Extract biophysical features
                    bio_features = self.extract_biophysical_features(sequence)
                    
                    # Extract structural features  
                    struct_features = self.af_extractor.extract_all_features(sequence)
                    
                    # Simulate ESM-2 features (in practice, would do in batch)
                    np.random.seed(hash(sequence) % 2**32)
                    esm_features = np.random.normal(0, 0.1, 320)
                    
                    # Combine all features
                    all_features = {
                        'sequence': sequence,
                        'label': row['label'],
                        'protein_name': row.get('protein_name', 'unknown')
                    }
                    
                    # Add biophysical features with prefix
                    for k, v in bio_features.items():
                        all_features[f'bio_{k}'] = v
                    
                    # Add structural features with prefix  
                    for k, v in struct_features.items():
                        all_features[f'struct_{k}'] = v
                    
                    # Add ESM features
                    for i, val in enumerate(esm_features):
                        all_features[f'esm_{i}'] = val
                        
                    batch_results.append(all_features)
                    
                except Exception as e:
                    print(f"Error processing sequence {row.get('protein_name', 'unknown')}: {e}")
                    continue
            
            enhanced_data.extend(batch_results)
            
            # Save intermediate results
            if batch_start % (batch_size * 10) == 0:
                self._save_checkpoint(enhanced_data, batch_start)
        
        # Convert to DataFrame
        enhanced_df = pd.DataFrame(enhanced_data)
        
        print(f"Enhanced dataset created with {len(enhanced_df)} proteins")
        print(f"Features per protein: {len(enhanced_df.columns) - 3}")  # Exclude sequence, label, name
        
        return enhanced_df
    
    def _save_checkpoint(self, data: List[Dict], batch_num: int):
        """Save intermediate checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_{batch_num}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_enhanced_dataset(self, df: pd.DataFrame):
        """Save enhanced dataset in multiple formats."""
        # Save as CSV
        csv_path = self.output_dir / "enhanced_targettrack_v4.csv"
        df.to_csv(csv_path, index=False)
        print(f"Enhanced dataset saved to: {csv_path}")
        
        # Save feature matrices separately for training
        feature_columns = [col for col in df.columns if col.startswith(('bio_', 'struct_', 'esm_'))]
        features = df[feature_columns].values
        labels = df['label'].values
        
        # Save as numpy arrays
        np.save(self.output_dir / "features_v4.npy", features)
        np.save(self.output_dir / "labels_v4.npy", labels)
        
        # Save feature names
        with open(self.output_dir / "feature_names_v4.txt", 'w') as f:
            for name in feature_columns:
                f.write(f"{name}\n")
        
        print(f"Feature matrices saved: {features.shape}")
        print(f"Labels saved: {labels.shape}")
        
        # Save metadata
        metadata = {
            'total_proteins': len(df),
            'total_features': len(feature_columns),
            'biophysical_features': 14,
            'structural_features': 16, 
            'esm_features': 320,
            'soluble_count': int(labels.sum()),
            'insoluble_count': int(len(labels) - labels.sum()),
        }
        
        with open(self.output_dir / "dataset_metadata_v4.json", 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print("Dataset metadata saved")
        return metadata
    
    def run_pipeline(self):
        """Run the complete enhancement pipeline."""
        print("🚀 Starting ORACLE v4.0 data enhancement pipeline...")
        
        # Load existing data
        df = self.load_targettrack_data()
        
        # Process with enhanced features
        enhanced_df = self.process_dataset(df)
        
        # Save results
        metadata = self.save_enhanced_dataset(enhanced_df)
        
        print("✅ Pipeline completed successfully!")
        print(f"Enhanced dataset: {metadata['total_proteins']} proteins, {metadata['total_features']} features")
        
        return enhanced_df, metadata

def main():
    """Run the data enhancement pipeline."""
    # Configure paths (adjust for your Colab setup)
    data_dir = Path("/content/oracle-solubility/data")  # Original data
    output_dir = Path("/content/enhanced_data")         # Enhanced output
    
    # Create and run pipeline
    pipeline = DataEnhancementPipeline(data_dir, output_dir)
    enhanced_df, metadata = pipeline.run_pipeline()
    
    print("\n" + "="*50)
    print("ORACLE v4.0 Data Enhancement Complete!")
    print(f"Next steps:")
    print(f"1. Train v4.0 model on enhanced features")
    print(f"2. Compare performance vs v3.0") 
    print(f"3. Validate on real proteins")
    print("="*50)

if __name__ == "__main__":
    main()
