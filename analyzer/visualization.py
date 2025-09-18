"""
Visualization module for analysis results
분석 결과 시각화 모듈 (eval_summary.py의 시각화 기능 기반)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class VisualizationManager:
    """
    분석 결과 시각화 관리 클래스
    eval_summary.py의 시각화 기능을 모듈화
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize Visualization Manager
        
        Args:
            output_dir (str): Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Labels for display
        self.labels_display = ['MAL', 'SYS', 'IND', 'VUL', 'ORG']
    
    def plot_cdf_precision_recall(
        self,
        doc_stats_dict: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None
    ):
        """
        Plot CDF of precision and recall for multiple models
        
        Args:
            doc_stats_dict (Dict): Document statistics for each model
            save_path (Optional[str]): Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        model_colors = {
            'gpt-4o-mini': 'green',
            'gpt-3.5-turbo': 'orange',
            'model': 'blue'
        }
        
        for i, metric in enumerate(['precision', 'recall']):
            ax = axes[i]
            
            for model_name, df_doc in doc_stats_dict.items():
                if metric in df_doc.columns:
                    values = df_doc[metric].dropna().values
                    if len(values) > 0:
                        sorted_vals = np.sort(values)
                        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                        
                        color = model_colors.get(model_name, 'blue')
                        ax.plot(sorted_vals, cdf, label=model_name, color=color, linewidth=2)
            
            ax.set_xlabel("Score", fontsize=10)
            if i == 0:
                ax.set_ylabel("Cumulative Probability", fontsize=10)
            ax.set_title(metric.capitalize(), fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=9, loc='lower right', title='Model', title_fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "cdf_precision_recall.pdf", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_frequency_based_entity_similarity(
        self,
        label_freq_stats_dict: Dict[str, Dict[str, Dict[str, int]]],
        save_path: Optional[str] = None
    ):
        """
        Plot frequency-based entity similarity for multiple models
        
        Args:
            label_freq_stats_dict (Dict): Label frequency statistics for each model
            save_path (Optional[str]): Path to save the figure
        """
        fig, axes = plt.subplots(1, len(label_freq_stats_dict), figsize=(4*len(label_freq_stats_dict), 4), sharey=True)
        
        if len(label_freq_stats_dict) == 1:
            axes = [axes]
        
        for ax, (model_name, label_freq_stats) in zip(axes, label_freq_stats_dict.items()):
            # Calculate percentages
            true_counts = [label_freq_stats[label]["y_true"] for label in ["Malware", "System", "Indicator", "Vulnerability", "Organization"]]
            pred_counts = [label_freq_stats[label]["y_pred"] for label in ["Malware", "System", "Indicator", "Vulnerability", "Organization"]]
            
            true_total = sum(true_counts)
            pred_total = sum(pred_counts)
            
            true_pct = [x / true_total * 100 if true_total > 0 else 0 for x in true_counts]
            pred_pct = [x / pred_total * 100 if pred_total > 0 else 0 for x in pred_counts]
            
            # Plot
            ax.plot(self.labels_display, true_pct, marker='o', linestyle='-', 
                   label='Original', linewidth=2, markersize=6)
            ax.plot(self.labels_display, pred_pct, marker='s', linestyle='--', 
                   label='Summary', linewidth=2, markersize=6)
            
            ax.set_title(model_name, fontsize=12)
            ax.set_xlabel("Entity Label", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        axes[0].set_ylabel("Entity Frequency (%)", fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "frequency_based_entity_similarity.pdf", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison_metrics(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['precision', 'recall', 'f1'],
        save_path: Optional[str] = None
    ):
        """
        Plot model comparison metrics
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
            metrics (List[str]): Metrics to plot
            save_path (Optional[str]): Path to save the figure
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            # Find columns related to this metric
            metric_cols = [col for col in comparison_df.columns if metric in col and 'overall' in col]
            
            if metric_cols:
                col = metric_cols[0]
                ax.bar(comparison_df['model'], comparison_df[col])
                ax.set_title(f'Overall {metric.capitalize()}', fontsize=12)
                ax.set_ylabel(metric.capitalize(), fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for i, v in enumerate(comparison_df[col]):
                    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "model_comparison_metrics.pdf", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_label_wise_performance(
        self,
        metrics_dict: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ):
        """
        Plot label-wise performance for multiple models
        
        Args:
            metrics_dict (Dict): Metrics for each model
            save_path (Optional[str]): Path to save the figure
        """
        labels = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
        metrics = ['precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x = np.arange(len(labels))
        width = 0.35
        
        models = list(metrics_dict.keys())
        colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(models)]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for j, (model_name, model_metrics) in enumerate(metrics_dict.items()):
                values = []
                for label in labels:
                    if 'label_metrics' in model_metrics and label in model_metrics['label_metrics']:
                        values.append(model_metrics['label_metrics'][label][metric])
                    else:
                        values.append(0)
                
                offset = width * (j - len(models)/2 + 0.5)
                ax.bar(x + offset, values, width, label=model_name, color=colors[j % len(colors)])
            
            ax.set_xlabel('Entity Labels')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'Label-wise {metric.capitalize()}')
            ax.set_xticks(x)
            ax.set_xticklabels(self.labels_display, rotation=45)
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "label_wise_performance.pdf", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_qa_performance_heatmap(
        self,
        metrics_dict: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ):
        """
        Plot QA performance as heatmap
        
        Args:
            metrics_dict (Dict): QA metrics for each model
            save_path (Optional[str]): Path to save the figure
        """
        # Prepare data for heatmap
        models = list(metrics_dict.keys())
        labels = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
        
        # Create matrix for F1 scores
        f1_matrix = []
        
        for model_name in models:
            model_f1s = []
            model_metrics = metrics_dict[model_name]
            
            for label in labels:
                if 'label_metrics' in model_metrics and label in model_metrics['label_metrics']:
                    model_f1s.append(model_metrics['label_metrics'][label]['f1'])
                else:
                    model_f1s.append(0)
            
            f1_matrix.append(model_f1s)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(f1_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(self.labels_display)
        ax.set_yticklabels(models)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('F1 Score', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{f1_matrix[i][j]:.3f}', 
                             ha="center", va="center", color="black")
        
        ax.set_title("QA Performance Heatmap (F1 Scores)")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "qa_performance_heatmap.pdf", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_failure_analysis(
        self,
        failure_analysis: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Plot failure analysis results
        
        Args:
            failure_analysis (Dict): Failure analysis data
            save_path (Optional[str]): Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Failure counts by label
        labels = list(failure_analysis['label_failure_counts'].keys())
        counts = list(failure_analysis['label_failure_counts'].values())
        
        ax1.bar(labels, counts, color='lightcoral')
        ax1.set_title('Failure Counts by Label')
        ax1.set_xlabel('Entity Labels')
        ax1.set_ylabel('Number of Failures')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # Plot 2: Total failures pie chart
        total_failures = failure_analysis['total_failures']
        if total_failures > 0:
            sizes = list(failure_analysis['label_failure_counts'].values())
            labels_pie = [f"{label}\n({count})" for label, count in zip(labels, sizes)]
            
            ax2.pie(sizes, labels=labels_pie, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Failure Distribution\n(Total: {total_failures})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "failure_analysis.pdf", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_report_figure(
        self,
        doc_stats_dict: Dict[str, pd.DataFrame],
        label_freq_stats_dict: Dict[str, Dict[str, Dict[str, int]]],
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive summary report figure
        
        Args:
            doc_stats_dict (Dict): Document statistics
            label_freq_stats_dict (Dict): Label frequency statistics
            save_path (Optional[str]): Path to save the figure
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: CDF (top row, spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        
        model_colors = {'gpt-4o-mini': 'green', 'gpt-3.5-turbo': 'orange'}
        
        for model_name, df_doc in doc_stats_dict.items():
            if 'precision' in df_doc.columns:
                values = df_doc['precision'].dropna().values
                if len(values) > 0:
                    sorted_vals = np.sort(values)
                    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                    color = model_colors.get(model_name, 'blue')
                    ax1.plot(sorted_vals, cdf, label=f'{model_name} Precision', color=color, linewidth=2)
        
        ax1.set_xlabel("Score")
        ax1.set_ylabel("Cumulative Probability")
        ax1.set_title("Model Performance Comparison (CDF)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        # Plot 2: Frequency comparison (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        
        if len(label_freq_stats_dict) >= 1:
            model_name = list(label_freq_stats_dict.keys())[0]
            label_freq_stats = label_freq_stats_dict[model_name]
            
            true_counts = [label_freq_stats[label]["y_true"] for label in ["Malware", "System", "Indicator", "Vulnerability", "Organization"]]
            pred_counts = [label_freq_stats[label]["y_pred"] for label in ["Malware", "System", "Indicator", "Vulnerability", "Organization"]]
            
            true_total = sum(true_counts)
            pred_total = sum(pred_counts)
            
            true_pct = [x / true_total * 100 if true_total > 0 else 0 for x in true_counts]
            pred_pct = [x / pred_total * 100 if pred_total > 0 else 0 for x in pred_counts]
            
            ax2.plot(self.labels_display, true_pct, marker='o', linestyle='-', label='Original')
            ax2.plot(self.labels_display, pred_pct, marker='s', linestyle='--', label='Summary')
            ax2.set_title(f'Entity Frequency - {model_name}')
            ax2.set_xlabel("Entity Label")
            ax2.set_ylabel("Frequency (%)")
            ax2.legend()
            ax2.grid(True, alpha=0.5)
        
        # Plot 3: Performance metrics (bottom right)
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate average metrics across models
        models = list(doc_stats_dict.keys())
        metrics = ['precision', 'recall', 'f1']
        
        if models:
            avg_metrics = {}
            for metric in metrics:
                avg_metrics[metric] = []
                for model_name in models:
                    df = doc_stats_dict[model_name]
                    if metric in df.columns:
                        avg_val = df[metric].mean()
                        avg_metrics[metric].append(avg_val)
                    else:
                        avg_metrics[metric].append(0)
            
            x = np.arange(len(models))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                ax3.bar(x + i*width, avg_metrics[metric], width, label=metric.capitalize())
            
            ax3.set_xlabel('Models')
            ax3.set_ylabel('Average Score')
            ax3.set_title('Average Performance Metrics')
            ax3.set_xticks(x + width)
            ax3.set_xticklabels(models)
            ax3.legend()
            ax3.grid(True, axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "summary_report.pdf", dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Test the visualization manager
    viz = VisualizationManager()
    
    # Example data for testing
    example_doc_stats = {
        'model1': pd.DataFrame({
            'precision': [0.8, 0.7, 0.9],
            'recall': [0.6, 0.8, 0.7],
            'f1': [0.7, 0.75, 0.8]
        }),
        'model2': pd.DataFrame({
            'precision': [0.75, 0.8, 0.85],
            'recall': [0.7, 0.75, 0.8],
            'f1': [0.72, 0.77, 0.82]
        })
    }
    
    print("Visualization manager initialized.")
    print(f"Output directory: {viz.output_dir}")
    print("Visualization test completed.")
