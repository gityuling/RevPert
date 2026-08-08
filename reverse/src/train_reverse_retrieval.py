"""Archived dual-encoder trainer is not part of the RevPert release.

Train the gallery-native residual model with:
  python reverse/scripts/run_gallery_dual_g2.py --cell_line hepg2 --seed 1
"""

raise ImportError(
    "Prototype dual-encoder training is archived and not shipped in this release. "
    "Use reverse/scripts/run_gallery_dual_g2.py for RevPert."
)
