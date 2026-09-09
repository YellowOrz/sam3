"""Export a text-initialized or random SAM3 target feature (one-time setup)."""

import argparse
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--init", choices=("text", "random"), default="text")
    parser.add_argument("--random-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bpe-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.reference_text.strip() or args.random_std <= 0:
        parser.error("reference-text must be nonempty and random-std must be positive")
    if Path(args.output).exists():
        parser.error("output already exists; choose a new output path")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    import torch
    from sam3.model.learned_prompt import checkpoint_sha256, LearnedPrompt
    from sam3.model_builder import _create_text_encoder, _DEFAULT_BPE_PATH

    # Construct only the text tower, not the visual model or video tracker.
    encoder = _create_text_encoder(args.bpe_path or _DEFAULT_BPE_PATH)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint = checkpoint.get("model", checkpoint)
    prefix = "detector.backbone.language_backbone."
    state = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    encoder.load_state_dict(state, strict=True)
    del checkpoint, state
    encoder.to(args.device).eval().requires_grad_(False)
    with torch.no_grad():
        mask, features, _ = encoder([args.reference_text], device=args.device)
    features, mask = features[:, 0].float().cpu(), mask[0].cpu()
    if args.init == "random":
        generator = torch.Generator().manual_seed(args.seed)
        features = torch.randn(features.shape, generator=generator) * args.random_std
    prompt = LearnedPrompt(
        features,
        mask,
        args.target_id,
        metadata={
            "base_checkpoint_sha256": checkpoint_sha256(args.checkpoint),
            "reference_text": args.reference_text,
            "initialization": args.init,
            "random_std": args.random_std if args.init == "random" else None,
            "seed": args.seed,
        },
    )
    prompt.save(args.output)
    print(f"Saved {args.output}: {prompt.features.numel()} trainable values")


if __name__ == "__main__":
    main()
