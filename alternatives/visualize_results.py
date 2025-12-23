"""
Visualization script for RAG evaluation results
Creates comparison charts between baseline and enhanced versions
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(baseline_file: str, enhanced_file: str):
    """Load baseline and enhanced results"""
    baseline = json.loads(Path(baseline_file).read_text())
    enhanced = json.loads(Path(enhanced_file).read_text())
    return baseline, enhanced


def create_comparison_chart(baseline: dict, enhanced: dict, output_file: str = "comparison.png"):
    """Create side-by-side comparison chart"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RAG System Evaluation: Baseline vs Enhanced', fontsize=16, fontweight='bold')
    
    metrics = list(baseline["metrics"].keys())
    baseline_scores = [baseline["metrics"][m] for m in metrics]
    enhanced_scores = [enhanced["metrics"][m] for m in metrics]
    improvements = [((e - b) / b * 100) if b > 0 else 0 for b, e in zip(baseline_scores, enhanced_scores)]
    
    # Chart 1: Bar comparison
    ax1 = axes[0, 0]
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, enhanced_scores, width, label='Enhanced', color='#4ECDC4', alpha=0.8)
    
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Chart 2: Improvement percentage
    ax2 = axes[0, 1]
    colors = ['#2ECC71' if imp >= 30 else '#F39C12' if imp >= 0 else '#E74C3C' for imp in improvements]
    bars = ax2.barh(metrics, improvements, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Improvement (%)', fontweight='bold')
    ax2.set_title('Percentage Improvement', fontweight='bold')
    ax2.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
    ax2.axvline(x=30, color='green', linestyle='--', linewidth=2, label='Target (30%)')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax2.text(val + 2, i, f'{val:+.1f}%', va='center', fontweight='bold')
    
    # Chart 3: Radar chart
    ax3 = axes[1, 0]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    baseline_scores_radar = baseline_scores + [baseline_scores[0]]
    enhanced_scores_radar = enhanced_scores + [enhanced_scores[0]]
    angles += angles[:1]
    
    ax3 = plt.subplot(223, projection='polar')
    ax3.plot(angles, baseline_scores_radar, 'o-', linewidth=2, label='Baseline', color='#FF6B6B')
    ax3.fill(angles, baseline_scores_radar, alpha=0.15, color='#FF6B6B')
    ax3.plot(angles, enhanced_scores_radar, 'o-', linewidth=2, label='Enhanced', color='#4ECDC4')
    ax3.fill(angles, enhanced_scores_radar, alpha=0.15, color='#4ECDC4')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels([m.replace('_', '\n').title() for m in metrics], fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.set_title('Radar Comparison', fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.grid(True)
    
    # Chart 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    EVALUATION SUMMARY
    {'='*40}
    
    Test Size: {baseline['summary']['test_size']} questions
    
    BASELINE SYSTEM
    ───────────────
    Average Score: {baseline['summary']['avg_score']:.3f}
    
    ENHANCED SYSTEM
    ───────────────
    Average Score: {enhanced['summary']['avg_score']:.3f}
    
    OVERALL IMPROVEMENT
    ───────────────────
    Δ Average: {((enhanced['summary']['avg_score'] - baseline['summary']['avg_score']) / baseline['summary']['avg_score'] * 100):+.1f}%
    
    TARGET ACHIEVEMENT
    ──────────────────
    Metrics with ≥30% improvement: {sum(1 for imp in improvements if imp >= 30)}
    
    """
    
    target_met = any(imp >= 30 for imp in improvements)
    if target_met:
        summary_text += "    ✅ TARGET MET: At least one metric\n       improved by ≥30%\n"
        summary_text += f"\n    Best improvement:\n       {metrics[improvements.index(max(improvements))].replace('_', ' ').title()}: {max(improvements):+.1f}%"
    else:
        summary_text += "    ⚠️  TARGET NOT MET: No metric reached\n       30% improvement\n"
        summary_text += f"\n    Best improvement:\n       {metrics[improvements.index(max(improvements))].replace('_', ' ').title()}: {max(improvements):+.1f}%"
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    
    return fig


def create_detailed_report(baseline: dict, enhanced: dict, output_file: str = "detailed_report.txt"):
    """Create detailed text report"""
    
    metrics = list(baseline["metrics"].keys())
    
    report = []
    report.append("="*80)
    report.append("ADVANCED RAG SYSTEM EVALUATION - DETAILED REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Evaluation Date: {enhanced['timestamp']}")
    report.append(f"Test Questions: {baseline['summary']['test_size']}")
    report.append("")
    
    report.append("INDIVIDUAL METRIC ANALYSIS")
    report.append("-"*80)
    
    for metric in metrics:
        baseline_score = baseline["metrics"][metric]
        enhanced_score = enhanced["metrics"][metric]
        improvement = ((enhanced_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
        
        report.append("")
        report.append(f"{metric.upper().replace('_', ' ')}")
        report.append(f"  Baseline Score:      {baseline_score:.4f}")
        report.append(f"  Enhanced Score:      {enhanced_score:.4f}")
        report.append(f"  Absolute Change:     {enhanced_score - baseline_score:+.4f}")
        report.append(f"  Relative Change:     {improvement:+.2f}%")
        
        if improvement >= 30:
            report.append(f"  Status:              ✅ TARGET ACHIEVED (≥30%)")
        elif improvement >= 10:
            report.append(f"  Status:              📈 Significant improvement")
        elif improvement >= 0:
            report.append(f"  Status:              📊 Minor improvement")
        else:
            report.append(f"  Status:              📉 Decreased")
    
    report.append("")
    report.append("="*80)
    report.append("SUMMARY STATISTICS")
    report.append("="*80)
    report.append(f"Baseline Average:     {baseline['summary']['avg_score']:.4f}")
    report.append(f"Enhanced Average:     {enhanced['summary']['avg_score']:.4f}")
    report.append(f"Overall Improvement:  {((enhanced['summary']['avg_score'] - baseline['summary']['avg_score']) / baseline['summary']['avg_score'] * 100):+.2f}%")
    report.append("")
    
    improvements = [((enhanced["metrics"][m] - baseline["metrics"][m]) / baseline["metrics"][m] * 100) 
                   if baseline["metrics"][m] > 0 else 0 for m in metrics]
    
    best_idx = improvements.index(max(improvements))
    worst_idx = improvements.index(min(improvements))
    
    report.append(f"Best Performing Metric:  {metrics[best_idx].replace('_', ' ').title()} ({improvements[best_idx]:+.1f}%)")
    report.append(f"Worst Performing Metric: {metrics[worst_idx].replace('_', ' ').title()} ({improvements[worst_idx]:+.1f}%)")
    report.append("")
    
    target_met = any(imp >= 30 for imp in improvements)
    report.append("TARGET ACHIEVEMENT")
    report.append("-"*80)
    if target_met:
        report.append("✅ SUCCESS: Project target achieved!")
        report.append(f"   {sum(1 for imp in improvements if imp >= 30)} metric(s) improved by ≥30%")
    else:
        report.append("⚠️  WARNING: Project target not yet met")
        report.append(f"   Best improvement: {max(improvements):.1f}% (target: ≥30%)")
        report.append("   Recommendation: Additional iterations needed")
    
    report.append("")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    Path(output_file).write_text(report_text, encoding="utf-8")
    print(f"📄 Detailed report saved to: {output_file}")
    
    # Also print to console
    print("\n" + report_text)
    
    return report_text


def main():
    """Main execution"""
    
    if not Path("baseline_results.json").exists():
        print("❌ Error: baseline_results.json not found")
        print("   Run evaluate_baseline.py first!")
        return
    
    if not Path("enhanced_results.json").exists():
        print("❌ Error: enhanced_results.json not found")
        print("   Run evaluate_enhanced.py first!")
        return
    
    print("📊 Loading results...")
    baseline, enhanced = load_results("baseline_results.json", "enhanced_results.json")
    
    print("📈 Creating visualizations...")
    create_comparison_chart(baseline, enhanced)
    
    print("📝 Generating detailed report...")
    create_detailed_report(baseline, enhanced)
    
    print("\n✨ Analysis complete!")
    print("\nGenerated files:")
    print("  - comparison.png (visual comparison)")
    print("  - detailed_report.txt (comprehensive analysis)")


if __name__ == "__main__":
    main()
