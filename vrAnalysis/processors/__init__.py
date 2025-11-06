"""Processors module for vrAnalysis.

This module provides data processing pipelines that transform session data
into analysis-ready formats.
"""

from .spkmaps import SpkmapProcessor, SpkmapParams, Maps

__all__ = ["SpkmapProcessor", "SpkmapParams", "Maps"]
