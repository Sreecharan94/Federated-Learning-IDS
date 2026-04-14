# experiments/generate_reports.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# REMOVED plotly imports as they caused errors and weren't used in core functions shown
#import plotly.graph_objects as go
#import plotly.express as px
from datetime import datetime
import json
import yaml
from typing import Dict, List, Any, Optional # Add Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, config_path: str = "./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config.get('monitoring', {}).get('plot_dir', './outputs/results/'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def _plot_federated_training_progress(self, df: pd.DataFrame):
        """Plot federated training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy plot
        ax1.plot(df['round'], df['accuracy'], 'b-', linewidth=2, marker='o', markersize=6)
        ax1.set_title('Global Model Accuracy Over Rounds')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Loss plot
        ax2.plot(df['round'], df['loss'], 'r-', linewidth=2, marker='s', markersize=6)
        ax2.set_title('Global Model Loss Over Rounds')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "federated_training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_client_participation(self, df: pd.DataFrame):
        """Plot client participation over rounds"""
        plt.figure(figsize=(12, 6))
        plt.plot(df['round'], df['active_clients'], 'g-', linewidth=2, marker='^', markersize=6)
        plt.title('Active Clients Per Round')
        plt.xlabel('Round')
        plt.ylabel('Number of Active Clients')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "client_participation.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_round_performance(self, df: pd.DataFrame):
        """Plot round performance metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # ... plotting logic for round times, combined metrics, etc.
        plt.tight_layout()
        plt.savefig(self.output_dir / "round_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_attack_distribution(self, attack_dist: Dict[str, int]):
        """Plot attack type distribution"""
        if not attack_dist:
            return
        # ... plotting logic ...
        plt.savefig(self.output_dir / "attack_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_attack_timeline(self, attack_metrics: Dict[str, Any]):
        """Plot attack detection timeline"""
        # ... plotting logic ...
        plt.savefig(self.output_dir / "attack_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_federated_learning_report(self, training_history: Dict[str, List],
                                         label_mapping: Optional[Dict[int, str]] = None) -> str: # Make label_mapping Optional
        """Generate federated learning performance report"""
        logger.info("Generating federated learning report...")

        df = pd.DataFrame({
            'round': training_history['rounds'],
            'accuracy': training_history['global_accuracy'],
            'loss': training_history['global_loss'],
            'active_clients': training_history['active_clients'],
            'round_time': training_history['round_times']
        })

        # Create comprehensive report
        report_content = []
        report_content.append("# Federated Learning Performance Report\n")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Basic statistics
        report_content.append("## Training Statistics\n")
        report_content.append(f"- Total Rounds: {len(training_history['rounds'])}")
        report_content.append(f"- Final Accuracy: {df['accuracy'].iloc[-1]:.4f}")
        report_content.append(f"- Best Accuracy: {df['accuracy'].max():.4f}")
        report_content.append(f"- Final Loss: {df['loss'].iloc[-1]:.4f}")
        report_content.append(f"- Average Active Clients: {df['active_clients'].mean():.2f}")
        report_content.append(f"- Average Round Time: {df['round_time'].mean():.2f}s\n")

        # Convergence analysis
        report_content.append("## Convergence Analysis\n")
        accuracy_improvement = df['accuracy'].iloc[-1] - df['accuracy'].iloc[0]
        report_content.append(f"- Accuracy Improvement: {accuracy_improvement:+.4f}")
        report_content.append(f"- Convergence Rate: {(accuracy_improvement / len(df)):.6f} per round\n")

        # Generate plots
        self._plot_federated_training_progress(df)
        self._plot_client_participation(df)
        self._plot_round_performance(df)

        # Save report
        report_path = self.output_dir / "federated_learning_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))

        logger.info(f"Federated learning report saved to {report_path}")
        return str(report_path)


    def generate_attack_detection_report(self, attack_metrics: Dict[str, Any],
                                       label_mapping: Optional[Dict[int, str]] = None) -> str: # Make label_mapping Optional
        """Generate attack detection performance report"""
        logger.info("Generating attack detection report...")

        report_content = []
        report_content.append("# Attack Detection Performance Report\n")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Attack distribution
        attack_dist = attack_metrics.get('attack_types', {})
        if attack_dist:
            total_attacks = sum(attack_dist.values())

            report_content.append("## Attack Distribution\n")
            for attack_type, count in attack_dist.items():
                percentage = (count / total_attacks) * 100 if total_attacks > 0 else 0
                report_content.append(f"- {attack_type}: {count} ({percentage:.2f}%)")
            report_content.append(f"\nTotal Attacks Detected: {total_attacks}\n")

        # Generate plots
        self._plot_attack_distribution(attack_dist)
        self._plot_attack_timeline(attack_metrics)

        # Save report
        report_path = self.output_dir / "attack_detection_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))

        logger.info(f"Attack detection report saved to {report_path}")
        return str(report_path)

    # ... (rest of the class remains largely the same)

def test_report_generator():
    """Test the report generator"""
    generator = ReportGenerator()

    # Create mock training history
    training_history = {
        'rounds': list(range(1, 11)),
        'global_accuracy': [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.87],
        'global_loss': [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38, 0.35],
        'active_clients': [4, 4, 5, 4, 5, 4, 5, 4, 5, 4],
        'round_times': [15.2, 14.8, 16.1, 15.5, 14.9, 15.3, 16.0, 15.7, 14.6, 15.1]
    }

    # Generate federated learning report - Pass None for label_mapping if not applicable
    fl_report = generator.generate_federated_learning_report(training_history, label_mapping=None)
    print(f"Generated federated learning report: {fl_report}")

    # Create mock attack metrics
    attack_metrics = {
        'attack_types': {
            'Benign': 1500,
            'DDoS': 300,
            'DoS': 200,
            'Bot': 150,
            'Brute_Force': 100
        }
    }

    # Generate attack detection report - Pass None for label_mapping if not applicable
    attack_report = generator.generate_attack_detection_report(attack_metrics, label_mapping=None)
    print(f"Generated attack detection report: {attack_report}")

    # ... (rest of the test remains the same)
    print("Report generator test passed!")

if __name__ == "__main__":
    test_report_generator()