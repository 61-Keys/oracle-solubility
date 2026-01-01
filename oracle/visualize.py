"""
ORACLE Visualizations - Modern, Clean, Minimal Design
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .predictor import PredictionResult


class Visualizer:
    """Apple-inspired minimal and elegant visualizations for ORACLE predictions."""

    # === APPLE-INSPIRED DESIGN SYSTEM ===
    COLORS = {
        # Softer, more refined primary colors
        'primary': '#007AFF',        # iOS blue
        'primary_light': '#E5F1FF',
        'success': '#34C759',        # iOS green
        'success_light': '#E8F8EB',
        'warning': '#FF9500',        # iOS orange
        'warning_light': '#FFF4E5',
        'danger': '#FF3B30',         # iOS red
        'danger_light': '#FFE8E6',
        'purple': '#AF52DE',         # iOS purple
        'purple_light': '#F5EBFF',
        'pink': '#FF2D55',           # iOS pink
        'pink_light': '#FFE5EC',
        # Refined grays with better hierarchy
        'gray_50': '#FAFAFA',
        'gray_100': '#F5F5F7',       # Apple's light gray
        'gray_200': '#E8E8ED',
        'gray_300': '#D1D1D6',
        'gray_400': '#AEAEB2',
        'gray_500': '#8E8E93',
        'gray_600': '#636366',
        'gray_800': '#1D1D1F',       # Apple's near-black
        'white': '#FFFFFF',
        'bg': '#FBFBFD',             # Slightly warmer white
    }
    
    def visualize(self, result: "PredictionResult", style: str = "full"):
        """Create visualization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return
        
        # Apple-inspired typography and styling
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['SF Pro Display', 'SF Pro Text', '-apple-system',
                               'system-ui', 'BlinkMacSystemFont', 'Segoe UI',
                               'Helvetica Neue', 'Arial', 'sans-serif'],
            'font.size': 11,
            'font.weight': 400,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.labelweight': 500,
            'axes.titleweight': 600,
            'axes.titlesize': 13,
            'figure.dpi': 110,  # Retina-like quality
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
        """Full dashboard layout with Apple-inspired design."""
        import matplotlib.patches as patches
        from matplotlib.gridspec import GridSpec
        from matplotlib.patheffects import withStroke

        # More generous layout with better spacing
        fig = plt.figure(figsize=(14, 9), facecolor=self.COLORS['bg'])
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                      left=0.06, right=0.94, top=0.88, bottom=0.08)

        score = result.solubility_score
        color, light_color = self._get_color(score)

        # Title with better hierarchy
        fig.text(0.06, 0.95, 'ORACLE', fontsize=26, fontweight=700,
                 color=self.COLORS['gray_800'], letterspace=0.5)
        fig.text(0.06, 0.91, 'Protein Solubility Prediction', fontsize=12,
                 fontweight=500, color=self.COLORS['gray_500'])
        
        # === SCORE CARD (top left) ===
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        ax1.set_facecolor(self.COLORS['bg'])

        # Card with subtle shadow (Apple-style)
        shadow = patches.FancyBboxPatch((-0.05, -0.15), 10.1, 10.2,
                                       boxstyle="round,pad=0.05,rounding_size=0.5",
                                       facecolor=self.COLORS['gray_200'],
                                       edgecolor='none', alpha=0.3, zorder=0)
        ax1.add_patch(shadow)

        card = patches.FancyBboxPatch((0, 0), 10, 10,
                                     boxstyle="round,pad=0.05,rounding_size=0.5",
                                     facecolor=self.COLORS['white'],
                                     edgecolor='none', zorder=1)
        ax1.add_patch(card)

        # Larger, softer score circle
        circle_bg = plt.Circle((3, 5.2), 2.5, facecolor=light_color,
                              edgecolor='none', alpha=0.5)
        ax1.add_patch(circle_bg)

        circle = plt.Circle((3, 5.2), 2.3, facecolor='none',
                          edgecolor=color, linewidth=5, alpha=0.9)
        ax1.add_patch(circle)

        # Score text with better typography
        ax1.text(3, 5.5, f"{score:.0%}", fontsize=38, fontweight=700,
                 ha='center', va='center', color=color)
        ax1.text(3, 3.7, "Solubility", fontsize=11, fontweight=500,
                 ha='center', va='center', color=self.COLORS['gray_500'])

        # Status with refined styling
        label = "SOLUBLE" if result.soluble else "INSOLUBLE"
        ax1.text(7, 6.8, label, fontsize=17, fontweight=700,
                 ha='center', color=color, letterspace=0.3)
        ax1.text(7, 6, f"{result.confidence:.0%} confidence", fontsize=10.5,
                 fontweight=500, ha='center', color=self.COLORS['gray_500'])

        # Length with better spacing
        ax1.text(7, 3.8, f"{len(result.sequence)}", fontsize=28, fontweight=700,
                 ha='center', color=self.COLORS['gray_800'])
        ax1.text(7, 2.7, "amino acids", fontsize=10.5, fontweight=500,
                 ha='center', color=self.COLORS['gray_500'])
        
        # === PROPERTIES CARD (top right) ===
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        ax2.set_facecolor(self.COLORS['bg'])

        # Card with shadow
        shadow2 = patches.FancyBboxPatch((-0.05, -0.15), 10.1, 10.2,
                                        boxstyle="round,pad=0.05,rounding_size=0.5",
                                        facecolor=self.COLORS['gray_200'],
                                        edgecolor='none', alpha=0.3, zorder=0)
        ax2.add_patch(shadow2)

        card2 = patches.FancyBboxPatch((0, 0), 10, 10,
                                      boxstyle="round,pad=0.05,rounding_size=0.5",
                                      facecolor=self.COLORS['white'],
                                      edgecolor='none', zorder=1)
        ax2.add_patch(card2)

        ax2.text(0.6, 9.2, "Sequence Properties", fontsize=13, fontweight=600,
                 color=self.COLORS['gray_800'])

        props = [
            ('Hydrophobicity', result.features.get('Hydrophobicity', 0), self.COLORS['primary']),
            ('Charged', result.features.get('Charged', 0), self.COLORS['purple']),
            ('Disorder Prone', result.features.get('Disorder Prone', 0), self.COLORS['warning']),
            ('Aggregation', result.features.get('Aggregation Prone', 0), self.COLORS['danger']),
            ('Polar', result.features.get('Polar', 0), self.COLORS['success']),
        ]

        # More generous spacing between items
        for i, (name, val, col) in enumerate(props):
            y = 7.6 - i * 1.5
            ax2.text(0.6, y, name, fontsize=10.5, fontweight=500,
                    color=self.COLORS['gray_600'], va='center')
            ax2.text(9.4, y, f"{val:.0%}", fontsize=10.5, fontweight=600,
                     color=self.COLORS['gray_800'], ha='right', va='center')

            # Softer, more elegant bars
            bar_bg = patches.FancyBboxPatch((4.5, y - 0.22), 4, 0.44,
                                            boxstyle="round,rounding_size=0.22",
                                            facecolor=self.COLORS['gray_100'],
                                            edgecolor='none', alpha=0.6)
            ax2.add_patch(bar_bg)

            # Bar fill with subtle gradient effect
            bar_fill = patches.FancyBboxPatch((4.5, y - 0.22), 4 * min(val / 0.5, 1), 0.44,
                                              boxstyle="round,rounding_size=0.22",
                                              facecolor=col, edgecolor='none', alpha=0.85)
            ax2.add_patch(bar_fill)
        
        # === FEATURE BARS (bottom left) ===
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(self.COLORS['white'])

        features = [
            ('Hydrophobic', result.features.get('Hydrophobicity', 0), self.COLORS['primary']),
            ('Aggregation', result.features.get('Aggregation Prone', 0), self.COLORS['danger']),
            ('Disorder', result.features.get('Disorder Prone', 0), self.COLORS['warning']),
            ('Aromatic', result.features.get('Aromatic', 0), self.COLORS['pink']),
            ('Charged', result.features.get('Charged', 0), self.COLORS['purple']),
            ('Polar', result.features.get('Polar', 0), self.COLORS['success']),
        ]

        names = [f[0] for f in features]
        values = [f[1] for f in features]
        colors = [f[2] for f in features]

        y_pos = np.arange(len(names))
        # Rounded bars with better spacing
        bars = ax3.barh(y_pos, values, color=colors, height=0.65,
                       edgecolor='none', alpha=0.85)

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names, fontsize=10.5, fontweight=500)
        ax3.set_xlim(0, 0.58)
        ax3.set_xlabel('Fraction', fontsize=10.5, fontweight=500,
                      color=self.COLORS['gray_500'], labelpad=8)
        ax3.set_title('Feature Analysis', fontsize=13, fontweight=600,
                     loc='left', pad=12, color=self.COLORS['gray_800'])
        ax3.tick_params(axis='both', length=0, colors=self.COLORS['gray_500'])
        ax3.spines['left'].set_visible(False)
        ax3.spines['bottom'].set_color(self.COLORS['gray_300'])
        ax3.spines['bottom'].set_linewidth(0.8)

        # Value labels with better positioning
        for i, v in enumerate(values):
            ax3.text(v + 0.012, i, f'{v:.0%}', va='center',
                    fontsize=9.5, fontweight=500, color=self.COLORS['gray_600'])
        
        # === HYDROPATHY (bottom right) ===
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(self.COLORS['white'])

        from .features import FeatureExtractor
        fe = FeatureExtractor()
        positions, scores = fe.compute_hydropathy_profile(result.sequence, window=9)

        if positions:
            # Softer gradient fills
            ax4.fill_between(positions, scores, 0,
                            where=[s > 0 for s in scores],
                            color=self.COLORS['danger'], alpha=0.12,
                            linewidth=0)
            ax4.fill_between(positions, scores, 0,
                            where=[s <= 0 for s in scores],
                            color=self.COLORS['success'], alpha=0.12,
                            linewidth=0)
            # Refined line style
            ax4.plot(positions, scores, color=self.COLORS['primary'],
                    linewidth=2, alpha=0.8, solid_capstyle='round')
            ax4.axhline(y=0, color=self.COLORS['gray_300'],
                       linewidth=1, linestyle='-', alpha=0.5)

        ax4.set_xlabel('Position', fontsize=10.5, fontweight=500,
                      color=self.COLORS['gray_500'], labelpad=8)
        ax4.set_ylabel('Hydropathy', fontsize=10.5, fontweight=500,
                      color=self.COLORS['gray_500'], labelpad=8)
        ax4.set_title('Hydropathy Profile', fontsize=13, fontweight=600,
                     loc='left', pad=12, color=self.COLORS['gray_800'])
        ax4.tick_params(axis='both', length=0, colors=self.COLORS['gray_500'])
        ax4.spines['left'].set_color(self.COLORS['gray_300'])
        ax4.spines['bottom'].set_color(self.COLORS['gray_300'])
        ax4.spines['left'].set_linewidth(0.8)
        ax4.spines['bottom'].set_linewidth(0.8)
        ax4.grid(True, alpha=0.1, linewidth=0.5, color=self.COLORS['gray_300'])

        plt.tight_layout()
        plt.show()
    
    def _plot_minimal(self, result, plt):
        """Minimal card view with Apple-inspired elegance."""
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(4.5, 6), facecolor=self.COLORS['bg'])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        score = result.solubility_score
        color, light_color = self._get_color(score)

        # Card shadow
        shadow = patches.FancyBboxPatch((0.35, 0.35), 9.3, 11.4,
                                       boxstyle="round,rounding_size=0.6",
                                       facecolor=self.COLORS['gray_200'],
                                       edgecolor='none', alpha=0.25)
        ax.add_patch(shadow)

        # Card
        card = patches.FancyBboxPatch((0.4, 0.4), 9.2, 11.3,
                                     boxstyle="round,rounding_size=0.6",
                                     facecolor=self.COLORS['white'],
                                     edgecolor='none')
        ax.add_patch(card)

        # Title
        ax.text(5, 10.8, 'ORACLE', fontsize=11, fontweight=600,
                ha='center', color=self.COLORS['gray_400'], letterspace=0.5)

        # Large score with refined typography
        ax.text(5, 7.5, f"{score:.0%}", fontsize=56, fontweight=700,
                ha='center', va='center', color=color)

        # Label
        label = "SOLUBLE" if result.soluble else "INSOLUBLE"
        ax.text(5, 5.2, label, fontsize=15, fontweight=700,
                ha='center', color=color, letterspace=0.4)

        # Confidence bar with better styling
        bar_bg = patches.FancyBboxPatch((1.8, 3.5), 6.4, 0.55,
                                        boxstyle="round,rounding_size=0.28",
                                        facecolor=self.COLORS['gray_100'],
                                        edgecolor='none', alpha=0.6)
        ax.add_patch(bar_bg)

        bar_fill = patches.FancyBboxPatch((1.8, 3.5), 6.4 * result.confidence, 0.55,
                                          boxstyle="round,rounding_size=0.28",
                                          facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(bar_fill)

        ax.text(5, 2.6, f"{result.confidence:.0%} confidence", fontsize=10,
                fontweight=500, ha='center', color=self.COLORS['gray_500'])

        ax.text(5, 1.4, f"{len(result.sequence)} amino acids", fontsize=10,
                fontweight=500, ha='center', color=self.COLORS['gray_400'])

        plt.tight_layout()
        plt.show()