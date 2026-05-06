"""
MLflow Experiment Tracking Integration

Provides comprehensive experiment tracking for DRDO-level reproducibility:
- Hyperparameter logging
- Metric tracking
- Model versioning
- Artifact storage
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Try to import MLflow, provide fallback if not installed
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Using local tracking fallback.")


class ExperimentTracker:
    """
    Unified experiment tracking with MLflow backend.
    Falls back to local JSON tracking if MLflow is unavailable.
    """

    def __init__(
        self,
        experiment_name: str = 'radar_threat_classification',
        tracking_uri: str = 'outputs/mlruns',
        use_mlflow: bool = True
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI or local directory
            use_mlflow: Whether to use MLflow (if available)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        self._run_data = {
            'params': {},
            'metrics': {},
            'artifacts': [],
            'tags': {}
        }

        if self.use_mlflow:
            self._setup_mlflow()
        else:
            self._setup_local()

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        logger.info(f"MLflow tracking configured: {self.tracking_uri}")
        logger.info(f"Experiment: {self.experiment_name}")

    def _setup_local(self):
        """Configure local tracking fallback."""
        self.local_dir = Path(self.tracking_uri) / self.experiment_name
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.local_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Local tracking configured: {self.run_dir}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start a new tracking run."""
        if self.use_mlflow:
            mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)
        else:
            if run_name:
                self.run_id = run_name
            if tags:
                self._run_data['tags'].update(tags)

        logger.info(f"Started run: {run_name or self.run_id}")

    def end_run(self, status: str = 'FINISHED'):
        """End the current run."""
        if self.use_mlflow:
            mlflow.end_run(status=status)
        else:
            self._save_local_run()

        logger.info(f"Ended run with status: {status}")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.use_mlflow:
            # Flatten nested dicts for MLflow
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
        else:
            self._run_data['params'].update(params)

        logger.debug(f"Logged {len(params)} parameters")

    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        self.log_params({key: value})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        else:
            for key, value in metrics.items():
                if key not in self._run_data['metrics']:
                    self._run_data['metrics'][key] = []
                self._run_data['metrics'][key].append({
                    'value': value,
                    'step': step,
                    'timestamp': datetime.utcnow().isoformat()
                })

        logger.debug(f"Logged metrics at step {step}: {metrics}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if self.use_mlflow:
            mlflow.log_artifact(local_path, artifact_path)
        else:
            self._run_data['artifacts'].append({
                'local_path': local_path,
                'artifact_path': artifact_path,
                'logged_at': datetime.utcnow().isoformat()
            })

        logger.info(f"Logged artifact: {local_path}")

    def log_model(
        self,
        model,
        artifact_path: str = 'model',
        registered_name: Optional[str] = None
    ):
        """Log a PyTorch model."""
        if self.use_mlflow:
            try:
                import mlflow.pytorch
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_name
                )
            except Exception as e:
                logger.warning(f"MLflow model logging failed: {e}")
                # Fallback to manual save
                import torch
                model_path = Path(self.tracking_uri) / 'models' / f'{artifact_path}.pt'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                self.log_artifact(str(model_path))
        else:
            import torch
            model_path = self.run_dir / f'{artifact_path}.pt'
            torch.save(model.state_dict(), model_path)
            self._run_data['artifacts'].append({
                'type': 'model',
                'path': str(model_path)
            })

        logger.info(f"Logged model: {artifact_path}")

    def log_config(self, config: Dict[str, Any]):
        """Log full configuration as artifact."""
        config_path = Path(self.tracking_uri) / 'configs' / f'{self.run_id}_config.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.log_artifact(str(config_path), 'config')
        self.log_params(self._flatten_dict(config, max_depth=2))

    def log_confusion_matrix(self, cm, class_names: List[str], step: Optional[int] = None):
        """Log confusion matrix as artifact."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Confusion Matrix (Step {step})' if step else 'Confusion Matrix')

        cm_path = self.run_dir if not self.use_mlflow else Path(self.tracking_uri) / 'temp'
        cm_path.mkdir(parents=True, exist_ok=True)
        cm_file = cm_path / f'confusion_matrix_step{step or 0}.png'

        plt.savefig(cm_file, dpi=150, bbox_inches='tight')
        plt.close()

        self.log_artifact(str(cm_file), 'confusion_matrices')

    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = '',
        sep: str = '.',
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and max_depth > 0:
                items.extend(
                    self._flatten_dict(v, new_key, sep, max_depth - 1).items()
                )
            else:
                # Convert to string for MLflow compatibility
                items.append((new_key, str(v) if not isinstance(v, (int, float, str, bool)) else v))
        return dict(items)

    def _save_local_run(self):
        """Save run data to local JSON file."""
        run_file = self.run_dir / 'run_data.json'
        with open(run_file, 'w') as f:
            json.dump(self._run_data, f, indent=2, default=str)

        logger.info(f"Saved run data to {run_file}")

    def get_run_id(self) -> str:
        """Get current run ID."""
        if self.use_mlflow:
            return mlflow.active_run().info.run_id if mlflow.active_run() else None
        return self.run_id


class ModelRegistry:
    """
    Model versioning and registry for production deployment.
    """

    def __init__(self, registry_dir: str = 'outputs/model_registry'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / 'registry.json'
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': {}, 'production': None}

    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a new model version.

        Returns:
            Version string (e.g., 'v1', 'v2')
        """
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {'versions': []}

        versions = self.registry['models'][model_name]['versions']
        version = f"v{len(versions) + 1}"

        version_info = {
            'version': version,
            'model_path': model_path,
            'metrics': metrics,
            'config': config,
            'tags': tags or {},
            'registered_at': datetime.utcnow().isoformat(),
            'stage': 'staging'
        }

        versions.append(version_info)
        self._save_registry()

        logger.info(f"Registered model {model_name} {version}")
        return version

    def promote_to_production(self, model_name: str, version: str):
        """Promote a model version to production."""
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} not found")

        versions = self.registry['models'][model_name]['versions']
        for v in versions:
            if v['version'] == version:
                v['stage'] = 'production'
                self.registry['production'] = {
                    'model_name': model_name,
                    'version': version,
                    'model_path': v['model_path'],
                    'promoted_at': datetime.utcnow().isoformat()
                }
                self._save_registry()
                logger.info(f"Promoted {model_name} {version} to production")
                return

        raise ValueError(f"Version {version} not found for {model_name}")

    def get_production_model(self) -> Optional[Dict]:
        """Get the current production model info."""
        return self.registry.get('production')

    def list_models(self) -> Dict:
        """List all registered models."""
        return self.registry['models']


def init_tracking(
    config: Dict[str, Any],
    experiment_name: Optional[str] = None
) -> ExperimentTracker:
    """
    Initialize experiment tracking from config.

    Args:
        config: Configuration dictionary
        experiment_name: Override experiment name

    Returns:
        Configured ExperimentTracker
    """
    tracking_config = config.get('tracking', {})

    tracker = ExperimentTracker(
        experiment_name=experiment_name or tracking_config.get('experiment_name', 'radar_classification'),
        tracking_uri=tracking_config.get('tracking_uri', 'outputs/mlruns'),
        use_mlflow=tracking_config.get('use_mlflow', True)
    )

    return tracker
