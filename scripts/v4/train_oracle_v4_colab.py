"""
ORACLE v4.0 Training Script for Google Colab
Trains enhanced model with structural features.
"""

# Install required packages (run in Colab)
COLAB_SETUP = """
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers fair-esm biotite scikit-learn matplotlib seaborn
!pip install colabfold[alphafold] --quiet

# Mount Google Drive for data persistence
from google.colab import drive
drive.mount('/content/drive')

# Clone the project
!git clone https://github.com/61-Keys/oracle-solubility.git /content/oracle-solubility

# Create working directories
!mkdir -p /content/enhanced_data
!mkdir -p /content/models_v4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ProteinDataset(Dataset):
    """Dataset class for protein solubility data."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class OracleNetV4(nn.Module):
    """Enhanced ORACLE v4.0 Neural Network."""
    
    def __init__(self, input_dim=350, dropout_rate=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 1
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 2
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        return self.network(x)

class OracleTrainerV4:
    """Training pipeline for ORACLE v4.0."""
    
    def __init__(self, data_dir="/content/enhanced_data", model_dir="/content/models_v4"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    def load_data(self):
        """Load enhanced dataset."""
        print("Loading enhanced dataset...")
        
        # Load feature matrices
        features_path = self.data_dir / "features_v4.npy"
        labels_path = self.data_dir / "labels_v4.npy"
        metadata_path = self.data_dir / "dataset_metadata_v4.json"
        
        if not all(p.exists() for p in [features_path, labels_path, metadata_path]):
            print("Enhanced data not found. Run data enhancement pipeline first.")
            return None, None, None
        
        features = np.load(features_path)
        labels = np.load(labels_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded dataset: {features.shape} features, {labels.shape} labels")
        print(f"Soluble: {metadata['soluble_count']}, Insoluble: {metadata['insoluble_count']}")
        
        return features, labels, metadata
    
    def prepare_data(self, features, labels, test_size=0.2, val_size=0.1):
        """Prepare data for training."""
        print("Preparing data splits...")
        
        # Normalize features
        self.scaler = StandardScaler()
        features_normalized = self.scaler.fit_transform(features)
        
        # Create dataset
        dataset = ProteinDataset(features_normalized, labels)
        
        # Calculate split sizes
        total_size = len(dataset)
        test_size_abs = int(total_size * test_size)
        val_size_abs = int(total_size * val_size)
        train_size_abs = total_size - test_size_abs - val_size_abs
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size_abs, val_size_abs, test_size_abs],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size=128):
        """Create data loaders."""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for features, labels in tqdm(train_loader, desc="Training", leave=False):
            features, labels = features.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc="Validating", leave=False):
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        return avg_loss, accuracy, f1
    
    def train_model(self, train_loader, val_loader, epochs=100, lr=1e-3, patience=10):
        """Train the model."""
        print(f"Training ORACLE v4.0 for {epochs} epochs...")
        
        # Initialize model
        input_dim = next(iter(train_loader))[0].shape[1]
        model = OracleNetV4(input_dim=input_dim).to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        
        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Early stopping and checkpointing
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': self.scaler,
                    'epoch': epoch,
                    'val_f1': val_f1,
                    'input_dim': input_dim
                }, self.model_dir / "oracle_v4_best.pt")
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        print(f"Training completed. Best validation F1: {best_val_f1:.4f}")
        return model
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model on test set."""
        print("Evaluating on test set...")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="Testing"):
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of soluble
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, 
                                   target_names=['Insoluble', 'Soluble']))
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1')
        axes[1, 0].set_title('Training F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning curves
        axes[1, 1].plot(self.history['val_acc'], label='Validation Accuracy', color='blue')
        axes[1, 1].plot(self.history['val_f1'], label='Validation F1', color='red')
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_training(self):
        """Run the complete training pipeline."""
        print("🚀 Starting ORACLE v4.0 training pipeline...")
        
        # Load data
        features, labels, metadata = self.load_data()
        if features is None:
            print("❌ Data loading failed. Run enhancement pipeline first.")
            return None
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data(features, labels)
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Train model
        model = self.train_model(train_loader, val_loader)
        
        # Load best model for evaluation
        checkpoint = torch.load(self.model_dir / "oracle_v4_best.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_results = self.evaluate_model(model, test_loader)
        
        # Plot results
        self.plot_training_history()
        
        # Save results
        results = {
            'metadata': metadata,
            'test_results': test_results,
            'training_history': self.history
        }
        
        with open(self.model_dir / "training_results_v4.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    serializable_results[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, np.ndarray):
                            serializable_results[k][k2] = v2.tolist()
                        else:
                            serializable_results[k][k2] = v2
                else:
                    serializable_results[k] = v
            json.dump(serializable_results, f, indent=2)
        
        print("✅ Training pipeline completed!")
        print(f"Best model saved to: {self.model_dir / 'oracle_v4_best.pt'}")
        print(f"Results saved to: {self.model_dir / 'training_results_v4.json'}")
        
        return model, test_results

def main():
    """Main training function for Colab."""
    print("🧬 ORACLE v4.0 Training in Google Colab")
    
    # Initialize trainer
    trainer = OracleTrainerV4()
    
    # Run training
    model, results = trainer.run_complete_training()
    
    if model and results:
        print("\n" + "="*60)
        print("ORACLE v4.0 Training Complete!")
        print(f"Final Test Accuracy: {results['accuracy']:.2%}")
        print(f"Final Test F1 Score: {results['f1']:.3f}")
        print(f"Final Test AUC-ROC: {results['auc']:.3f}")
        print("="*60)
        
        # Compare with v3.0 baseline
        print("\nComparison with v3.0:")
        print(f"v3.0 Test Accuracy: 66.8%")
        print(f"v4.0 Test Accuracy: {results['accuracy']:.1%}")
        improvement = (results['accuracy'] - 0.668) * 100
        print(f"Improvement: +{improvement:.1f} percentage points")

if __name__ == "__main__":
    # For Colab, first run the setup
    print("First run this setup code in Colab:")
    print(COLAB_SETUP)
    print("\nThen run main() to start training")
    
    # Uncomment to run immediately
    # main()
