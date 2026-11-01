"""Lumen Engine - the local image-processing backend of Lumen Studio.

Everything in this package runs on the local machine. No module performs
network access except :mod:`lumen_engine.models.manager`, and only when the
user explicitly requests a model download.
"""

__version__ = "1.4.1"
