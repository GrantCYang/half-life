"""HalfLife: forecasting the endpoint validity of remembered scene facts.

The core package is CPU-only. Transformer and tree-model dependencies are
imported only by their optional execution paths.
"""

from halflife.data.schema import QueryExample, TypedObservation

__all__ = ["QueryExample", "TypedObservation"]
__version__ = "0.2.0"
