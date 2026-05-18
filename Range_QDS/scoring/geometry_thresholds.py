"""Shared geometric thresholds for final-candidate diagnostics."""

FINAL_LENGTH_PRESERVATION_MIN = 0.75
FINAL_LENGTH_PRESERVATION_MAX = 1.20
FINAL_AVG_SED_RATIO_MAX_LOW_COMPRESSION = 2.00
FINAL_AVG_SED_RATIO_MAX_MEDIUM_COMPRESSION = 1.75
FINAL_AVG_SED_RATIO_MAX_DEFAULT = 1.50
FINAL_LOW_COMPRESSION_RATIO_CUTOFF = 0.01
FINAL_MEDIUM_COMPRESSION_RATIO_CUTOFF = 0.02


def max_sed_ratio_for_compression(compression_ratio: float) -> float:
    """Return the final geometry SED-ratio limit for a compression ratio."""
    ratio = float(compression_ratio)
    if ratio <= FINAL_LOW_COMPRESSION_RATIO_CUTOFF + 1e-12:
        return FINAL_AVG_SED_RATIO_MAX_LOW_COMPRESSION
    if ratio <= FINAL_MEDIUM_COMPRESSION_RATIO_CUTOFF + 1e-12:
        return FINAL_AVG_SED_RATIO_MAX_MEDIUM_COMPRESSION
    return FINAL_AVG_SED_RATIO_MAX_DEFAULT
