"""Compatibility stub.

Train RevPert with the gallery-native residual script instead:

  python reverse/scripts/run_gallery_dual_g2.py --cell_line hepg2 --seed 1
"""

raise ImportError(
    "Use reverse/scripts/run_gallery_dual_g2.py to train RevPert "
    "(gallery-native residual scoring)."
)
