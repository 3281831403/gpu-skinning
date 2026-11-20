"""Public skinning API for the exercise."""

from .deform import (
    SkinError,
    apply_bone_matrix,
    blend_shapes,
    inverse_bind,
    normalize_weights,
    skin_batch,
    skin_normal,
    skin_vertex,
    validate_skin,
)

__all__ = [
    "SkinError",
    "apply_bone_matrix",
    "blend_shapes",
    "inverse_bind",
    "normalize_weights",
    "skin_batch",
    "skin_normal",
    "skin_vertex",
    "validate_skin",
]
