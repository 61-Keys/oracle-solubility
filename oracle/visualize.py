"""
ORACLE Visualizations - Modern, Clean, Minimal Design
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .predictor import PredictionResult


class Visualizer:
    """Modern, clean visualizations for ORACLE predictions."""
    
    # === DESIGN SYSTEM ===
    COLORS = {
        'primary': '#6366F1',
        'primary_light': '#E0E7FF',
        'success': '#10B981',
        'success_light': '#D1FAE5',
        'warning': '#F59E0B',
        'warning_light': '#FEF3C7',
        'danger': '#EF4444',
        'danger_light': '#FEE2E2',
        'gray_100': '#F3F4F6',
        'gray_200': '#E5E7EB',
        'gray_400': '#9CA3AF',
        'gray_500': '#6B7280',
        'gray_600': '#4B5563',
        'gray_800': '#1F2937',
        'white': '#FFFFFF',
        'bg': '#F9FAFB',
    }
    
    def visualize(self, result: "PredictionResult", style: str = "full"):
        """Create visualization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        if style == "minimal":
            self._plot_minimal(result, plt)
        else:
            self._plot_full(result, plt)
    
    def _get_color(self, score):
        if score >= 0.7:
            return self.COLORS['success'], self.COLORS['success_light']
        elif score >= 0.5:
            return self.COLORS['warning'], self.COLORS['warning_light']
        else:
            return self.COLORS['danger'], self.COLORS['danger_light']
    
    def _plot_full(self, result, plt):
        """Full dashboard layout."""
        import matplotlib.patches as patches
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(12, 8), facecolor=self.COLORS['bg'])
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25,
                      left=0.08, right=0.95, top=0.85, bottom=0.1)
        
        score = result.solubility_score
        color, light_color = self._get_color(score)
        
        # Title
        fig.text(0.08, 0.94, 'ORACLE', fontsize=22, fontweight='bold', 
                 color=self.COLORS['gray_800'])
        fig.text(0.08, 0.89, 'Protein Solubility Prediction', fontsize=11, 
                 color=self.COLORS['gray_500'])
        
        # === SCORE CARD (top left) ===
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        ax1.set_facecolor(self.COLORS['white'])
        
        # Add card border
        rect = patches.FancyBboxPatch((0.1, 0.1), 9.8, 9.8, 
                                       boxstyle="round,pad=0.02,rounding_size=0.3",
                                       facecolor=self.COLORS['white'],
                                       edgecolor=self.COLORS['gray_200'], linewidth=1)
        ax1.add_patch(rect)
        
        # Score circle
        circle = plt.Circle((3, 5), 2.2, facecolor=light_color, edgecolor=color, linewidth=4)
        ax1.add_patch(circle)
        
        # Score text
        ax1.text(3, 5.3, f"{score:.0%}", fontsize=32, fontweight='bold',
                 ha='center', va='center', color=self.COLORS['gray_800'])
        ax1.text(3, 3.8, "Solubility", fontsize=10, ha='center', va='center',
                 color=self.COLORS['gray_500'])
        
        # Status
        label = "SOLUBLE" if result.soluble else "INSOLUBLE"
        ax1.text(7, 6.5, label, fontsize=16, fontweight='bold', ha='center', color=color)
        ax1.text(7, 5.5, f"{result.confidence:.0%} confidence", fontsize=10, 
                 ha='center', color=self.COLORS['gray_500'])
        
        # Length
        ax1.text(7, 3.5, f"{len(result.sequence)}", fontsize=24, fontweight='bold',
                 ha='center', color=self.COLORS['gray_800'])
        ax1.text(7, 2.5, "amino acids", fontsize=10, ha='center', 
                 color=self.COLORS['gray_500'])
        
        # === PROPERTIES CARD (top right) ===
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        ax2.set_facecolor(self.COLORS['white'])
        
        rect2 = patches.FancyBboxPatch((0.1, 0.1), 9.8, 9.8,
                                        boxstyle="round,pad=0.02,rounding_size=0.3",
                                        facecolor=self.COLORS['white'],
                                        edgecolor=self.COLORS['gray_200'], linewidth=1)
        ax2.add_patch(rect2)
        
        ax2.text(0.5, 9, "Sequence Properties", fontsize=12, fontweight='bold',
                 color=self.COLORS['gray_800'])
        
        props = [
            ('Hydrophobicity', result.features.get('Hydrophobicity', 0), self.COLORS['primary']),
            ('Charged', result.features.get('Charged', 0), '#8B5CF6'),
            ('Disorder Prone', result.features.get('Disorder Prone', 0), self.COLORS['warning']),
            ('Aggregation', result.features.get('Aggregation Prone', 0), self.COLORS['danger']),
            ('Polar', result.features.get('Polar', 0), self.COLORS['success']),
        ]
        
        for i, (name, val, col) in enumerate(props):
            y = 7.5 - i * 1.4
            ax2.text(0.5, y, name, fontsize=10, color=self.COLORS['gray_600'], va='center')
            ax2.text(9.5, y, f"{val:.0%}", fontsize=10, fontweight='bold',
                     color=self.COLORS['gray_800'], ha='right', va='center')
            
            # Bar background
            bar_bg = patches.FancyBboxPatch((4, y - 0.25), 4.5, 0.5,
                                            boxstyle="round,rounding_size=0.15",
                                            facecolor=self.COLORS['gray_100'], edgecolor='none')
            ax2.add_patch(bar_bg)
            
            # Bar fill
            bar_fill = patches.FancyBboxPatch((4, y - 0.25), 4.5 * min(val / 0.5, 1), 0.5,
                                              boxstyle="round,rounding_size=0.15",
                                              facecolor=col, edgecolor='none', alpha=0.8)
            ax2.add_patch(bar_fill)
        
        # === FEATURE BARS (bottom left) ===
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(self.COLORS['white'])
        
        features = [
            ('Hydrophobic', result.features.get('Hydrophobicity', 0), self.COLORS['primary']),
            ('Aggregation', result.features.get('Aggregation Prone', 0), self.COLORS['danger']),
            ('Disorder', result.features.get('Disorder Prone', 0), self.COLORS['warning']),
            ('Aromatic', result.features.get('Aromatic', 0), '#EC4899'),
            ('Charged', result.features.get('Charged', 0), '#8B5CF6'),
            ('Polar', result.features.get('Polar', 0), self.COLORS['success']),
        ]
        
        names = [f[0] for f in features]
        values = [f[1] for f in features]
        colors = [f[2] for f in features]
        
        y_pos = np.arange(len(names))
        ax3.barh(y_pos, values, color=colors, height=0.6, edgecolor='none')
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names, fontsize=10)
        ax3.set_xlim(0, 0.55)
        ax3.set_xlabel('Fraction', fontsize=10, color=self.COLORS['gray_500'])
        ax3.set_title('Feature Analysis', fontsize=12, fontweight='bold', loc='left', pad=10)
        ax3.tick_params(axis='both', length=0)
        ax3.spines['left'].set_visible(False)
        ax3.spines['bottom'].set_color(self.COLORS['gray_200'])
        
        for i, v in enumerate(values):
            ax3.text(v + 0.01, i, f'{v:.0%}', va='center', fontsize=9, 
                     color=self.COLORS['gray_500'])
        
        # === HYDROPATHY (bottom right) ===
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(self.COLORS['white'])
        
        from .features import FeatureExtractor
        fe = FeatureExtractor()
        positions, scores = fe.compute_hydropathy_profile(result.sequence, window=9)
        
        if positions:
            ax4.fill_between(positions, scores, 0,
                            where=[s > 0 for s in scores],
                            color=self.COLORS['danger_light'], alpha=0.7)
            ax4.fill_between(positions, scores, 0,
                            where=[s <= 0 for s in scores],
                            color=self.COLORS['success_light'], alpha=0.7)
            ax4.plot(positions, scores, color=self.COLORS['gray_600'], linewidth=1.2)
            ax4.axhline(y=0, color=self.COLORS['gray_100'], linewidth=0.8)
        
        ax4.set_xlabel('Position', fontsize=10, color=self.COLORS['gray_500'])
        ax4.set_ylabel('Hydropathy', fontsize=10, color=self.COLORS['gray_500'])
        ax4.set_title('Hydropathy Profile', fontsize=12, fontweight='bold', loc='left', pad=10)
        ax4.tick_params(axis='both', length=0)
        ax4.spines['left'].set_color(self.COLORS['gray_200'])
        ax4.spines['bottom'].set_color(self.COLORS['gray_200'])
        
        plt.show()
    
    def _plot_minimal(self, result, plt):
        """Minimal card view."""
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(4, 5), facecolor=self.COLORS['bg'])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        score = result.solubility_score
        color, light_color = self._get_color(score)
        
        # Card
        card = patches.FancyBboxPatch((0.5, 0.5), 9, 11,
                                       boxstyle="round,rounding_size=0.4",
                                       facecolor=self.COLORS['white'],
                                       edgecolor=self.COLORS['gray_200'], linewidth=1)
        ax.add_patch(card)
        
        # Title
        ax.text(5, 10.5, 'ORACLE', fontsize=10, fontweight='bold', 
                ha='center', color=self.COLORS['gray_400'])
        
        # Score
        ax.text(5, 7.5, f"{score:.0%}", fontsize=48, fontweight='bold',
                ha='center', va='center', color=color)
        
        # Label
        label = "SOLUBLE" if result.soluble else "INSOLUBLE"
        ax.text(5, 5, label, fontsize=14, fontweight='bold', ha='center', color=color)
        
        # Confidence bar
        bar_bg = patches.FancyBboxPatch((2, 3.5), 6, 0.5,
                                        boxstyle="round,rounding_size=0.2",
                                        facecolor=self.COLORS['gray_100'], edgecolor='none')
        ax.add_patch(bar_bg)
        
        bar_fill = patches.FancyBboxPatch((2, 3.5), 6 * result.confidence, 0.5,
                                          boxstyle="round,rounding_size=0.2",
                                          facecolor=color, edgecolor='none')
        ax.add_patch(bar_fill)
        
        ax.text(5, 2.8, f"{result.confidence:.0%} confidence", fontsize=9,
                ha='center', color=self.COLORS['gray_500'])
        
        ax.text(5, 1.5, f"{len(result.sequence)} amino acids", fontsize=9,
                ha='center', color=self.COLORS['gray_400'])
        
        plt.tight_layout()
        plt.show()