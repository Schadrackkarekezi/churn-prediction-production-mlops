import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, roc_auc_score, average_precision_score
)
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("reports/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true, y_pred, y_proba=None, model_name="Model") -> Dict[str, Any]:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'precision_class_0': precision_score(y_true, y_pred, pos_label=0),
            'recall_class_0': recall_score(y_true, y_pred, pos_label=0),
            'f1_class_0': f1_score(y_true, y_pred, pos_label=0)
        }

        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn), 'false_positives': int(fp),
            'false_negatives': int(fn), 'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })

        self._log_metrics(metrics, model_name)
        return metrics

    def _log_metrics(self, metrics, model_name):
        logger.info(f"Results for {model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", save=True):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        axes[0].set_title(f'{model_name} - Counts')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        axes[1].set_title(f'{model_name} - Normalized')
        axes[1].set_ylabel('Actual')
        axes[1].set_xlabel('Predicted')

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
                       dpi=300, bbox_inches='tight')
        return fig

    def plot_roc_curve(self, y_true, y_proba, model_name="Model", save=True):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        if save:
            fig.savefig(self.output_dir / f"roc_curve_{model_name.lower().replace(' ', '_')}.png",
                       dpi=300, bbox_inches='tight')
        return fig

    def plot_precision_recall_curve(self, y_true, y_proba, model_name="Model", save=True):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR (AP = {ap:.4f})')
        baseline = y_true.sum() / len(y_true)
        ax.axhline(y=baseline, color='navy', linestyle='--', lw=2, label=f'Random ({baseline:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model_name} - PR Curve')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if save:
            fig.savefig(self.output_dir / f"pr_curve_{model_name.lower().replace(' ', '_')}.png",
                       dpi=300, bbox_inches='tight')
        return fig

    def plot_threshold_analysis(self, y_true, y_proba, model_name="Model", save=True) -> Tuple[plt.Figure, float]:
        thresholds = np.arange(0.1, 0.9, 0.05)
        metrics_list = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            metrics_list.append({
                'threshold': thresh,
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            })

        df = pd.DataFrame(metrics_list)
        optimal_thresh = df.loc[df['f1'].idxmax(), 'threshold']

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['threshold'], df['precision'], 'b-', label='Precision', lw=2)
        ax.plot(df['threshold'], df['recall'], 'g-', label='Recall', lw=2)
        ax.plot(df['threshold'], df['f1'], 'r-', label='F1', lw=2)
        ax.axvline(x=optimal_thresh, color='gray', linestyle='--', label=f'Optimal ({optimal_thresh:.2f})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} - Threshold Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            fig.savefig(self.output_dir / f"threshold_{model_name.lower().replace(' ', '_')}.png",
                       dpi=300, bbox_inches='tight')
        return fig, optimal_thresh

    def plot_feature_importance(self, model, feature_names, model_name="Model", top_n=20, save=True):
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model has no feature_importances_")
            return None

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(df['feature'], df['importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / f"importance_{model_name.lower().replace(' ', '_')}.png",
                       dpi=300, bbox_inches='tight')
        return fig

    def generate_report(self, y_true, y_pred, y_proba=None, model=None, feature_names=None, model_name="Model"):
        report = {
            'model_name': model_name,
            'metrics': self.evaluate(y_true, y_pred, y_proba, model_name),
            'classification_report': classification_report(y_true, y_pred),
            'plots': []
        }

        self.plot_confusion_matrix(y_true, y_pred, model_name)
        report['plots'].append('confusion_matrix')

        if y_proba is not None:
            self.plot_roc_curve(y_true, y_proba, model_name)
            self.plot_precision_recall_curve(y_true, y_proba, model_name)
            _, optimal = self.plot_threshold_analysis(y_true, y_proba, model_name)
            report['optimal_threshold'] = optimal
            report['plots'].extend(['roc_curve', 'pr_curve', 'threshold'])

        if model is not None and feature_names is not None:
            self.plot_feature_importance(model, feature_names, model_name)
            report['plots'].append('feature_importance')

        logger.info(f"\n{report['classification_report']}")
        return report
