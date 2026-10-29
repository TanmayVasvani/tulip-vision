"""Tulip Vision — visual quality inspection for toothbrush production.

Modules
    core        numpy-only anomaly math: coreset, memory bank, thresholds, metrics
    config      config.yaml loading, path resolution, per-SKU defaults
    data        EXIF-safe image loading, ROI crops, deterministic dataset splits
    features    frozen ImageNet backbone -> patch features (the torch code path)
    model       versioned offline model bundles: load, inspect, hot-swap
    train       training and evaluation (held-out goods, unseen defects)
    classifier  Phase-2 defect-type classifier on frozen features (numpy)
    vis         heatmap overlays and ROI previews
    camera      camera sources: USB/phone webcam, or folder replay for demos
    synth       synthetic toothbrush renderer for demos and tests
"""

__version__ = "2.0.0"
