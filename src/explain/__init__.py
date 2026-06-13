"""Explainability for the radar inspection framework (ML Grad-CAM + feature attribution)."""

from .gradcam import GradCAM, gradcam_on_rd_map
from .attribution import (
    integrated_gradients, occlusion_importance, ml_feature_importance,
)

__all__ = [
    "GradCAM", "gradcam_on_rd_map",
    "integrated_gradients", "occlusion_importance", "ml_feature_importance",
]
