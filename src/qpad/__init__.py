"""QPAD: Calibration-Set Composition Bias in INT8 Face Presentation Attack Detection.

Reusable library code for the QPAD notebooks. Notebooks orchestrate; the logic
lives here so it can be unit tested without the private OULU-NPU dataset.
"""

__version__ = "0.1.0"

CLASSES = ("bona_fide", "print", "replay")
ATTACK_CLASSES = ("print", "replay")
