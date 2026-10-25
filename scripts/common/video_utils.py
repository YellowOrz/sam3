"""Compatibility name for the public video utilities."""

import sys

from sam3 import video_io

sys.modules[__name__] = video_io
