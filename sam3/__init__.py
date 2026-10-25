# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe


def __getattr__(name):
    if name in __all__:
        from . import model_builder

        return getattr(model_builder, name)
    raise AttributeError(name)


__version__ = "0.1.0"

__all__ = ["build_sam3_image_model", "build_sam3_predictor"]
