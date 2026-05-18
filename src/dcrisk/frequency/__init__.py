"""Frequency layer: Negative-Binomial and Cox-process claim counts."""

from dcrisk.frequency.nb_gamma import Cox_intensity, sample_NB

__all__ = ["sample_NB", "Cox_intensity"]
