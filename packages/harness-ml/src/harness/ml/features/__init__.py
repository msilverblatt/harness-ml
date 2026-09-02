"""Feature system: schema, pairwise derivatives, resolver, and augmentation."""

from harness.ml.features.augmentation import augment_symmetric
from harness.ml.features.pairwise import generate_pairwise_derivatives
from harness.ml.features.resolver import FeatureResolver
from harness.ml.features.schema import FeatureDefinition, FeatureSet, FeatureType

__all__ = [
    "FeatureDefinition",
    "FeatureResolver",
    "FeatureSet",
    "FeatureType",
    "augment_symmetric",
    "generate_pairwise_derivatives",
]
