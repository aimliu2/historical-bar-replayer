"""Layer 4 trade hypotheses and the runtime extension registry."""

from .smc_3tf import SMC3TF


HYPOTHESIS_REGISTRY = {
    "SMC-3TF": SMC3TF,
}

__all__ = ["HYPOTHESIS_REGISTRY", "SMC3TF"]
