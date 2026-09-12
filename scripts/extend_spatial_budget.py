"""Explicit, non-destructive epoch-budget migration for spatial checkpoints.

Run with the frozen training implementation on PYTHONPATH. This does not
relax the trainer's strict resume contract or reset AdamW/RNG/progress.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from sam3.model.spatial_mask_adapter import SpatialMaskAdapter
from scripts.spatial_training_state import validate_state


def extend_budget(state, epochs, template, source_sha256, reason='Explicit completed-budget continuation; only epoch budget changes'):
    config = state['config']
    validate_state(state, config, template)
    if type(epochs) is not int or epochs <= config['epochs']:
        raise ValueError('New budget must strictly increase')
    if state['step'] != config['epochs'] * config['steps_per_epoch']:
        raise ValueError('Only completed budgets may be extended')
    if state['last_validation_step'] != state['step']:
        raise ValueError('Completed checkpoint must be validated')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('A nonempty extension reason is required')
    result = dict(state)
    result['config'] = dict(config, epochs=epochs)
    result['budget_extension'] = dict(source_sha256=source_sha256,
        old_epochs=config['epochs'], new_epochs=epochs,
        reason=reason)
    validate_state(result, result['config'], template)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--reason', default='Explicit completed-budget continuation; only epoch budget changes')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    state = torch.load(args.source, map_location='cpu', weights_only=True)
    template = SpatialMaskAdapter(bottleneck=state['config']['bottleneck']).state_dict()
    result = extend_budget(state, args.epochs, template, hashlib.sha256(raw).hexdigest(), reason=args.reason)
    with args.output.open('xb') as handle:
        torch.save(result, handle)
    loaded = torch.load(args.output, map_location='cpu', weights_only=True)
    validate_state(loaded, loaded['config'], template)
    print(f"Budget extended {state['config']['epochs']} -> {args.epochs}; step={state['step']}")


if __name__ == '__main__':
    main()
