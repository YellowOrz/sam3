"""Pure-Python checkpoint selection for one fixed, approved DexYCB validation set.

No model, dataset, file, CUDA or distributed operations occur here. The caller
must establish that ``validation_sha256`` identifies the approved DexYCB val
annotations; a digest or scope string alone cannot prove dataset provenance.

Call ``update_selection`` only once per completed epoch, with an optional
initial baseline at epoch 0. Steps and epochs must both strictly increase;
after resume skip the already-recorded validation instead of counting it twice.
``bad_epochs`` counts evaluated epochs, not skipped epochs or optimizer steps.

Actual best means any strictly higher macro Dice (ties retain the earlier
checkpoint). Early stopping uses a separate reference: only an improvement
strictly greater than min_delta resets patience. Small gains can accumulate
relative to that reference and are still eligible for an actual-best save.
Consequently an epoch can save a new best and exhaust patience at the same time.

The returned states are JSON-native checkpoint data. Restore replays the full
history and verifies every derived field, validation identity and policy. This
is consistency validation, not cryptographic authentication of a checkpoint.
Defaults are an experimental policy, not a guarantee of improvement.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral, Real


SCHEMA = "sam3-residual-selection-v1"
SCOPE = "dexycb_val"
METRIC_NAME = "dexycb_val/macro_miss_zero_dice"
SIDES = ("left_hand", "right_hand")
_FIXED_COUNTS = ("images", "queries", "targets") + tuple(
    f"{side}/{key}" for side in SIDES for key in ("positive_count", "absent_count")
)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must fit a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("validation_sha256 must be a lowercase SHA256 digest")
    return value


def _policy(patience, min_delta):
    patience = _integer(patience, "patience", minimum=1)
    min_delta = _finite(min_delta, "min_delta")
    if min_delta < 0:
        raise ValueError("min_delta must be nonnegative")
    return {"patience": patience, "min_delta": min_delta}


def _validation_metrics(metrics):
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a nonempty mapping")
    normalized = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metric keys must be nonempty strings")
        number = _finite(value, key)
        normalized[key] = int(value) if isinstance(value, Integral) else number

    required = set(_FIXED_COUNTS)
    for side in SIDES:
        required.update(f"{side}/{key}" for key in (
            "miss_zero_dice", "false_negative_count", "false_positive_count"))
    missing = required - normalized.keys()
    if missing:
        raise ValueError(f"Missing validation metrics: {sorted(missing)}")

    for key in _FIXED_COUNTS:
        normalized[key] = _integer(metrics[key], key)
    images = _integer(normalized["images"], "images", minimum=1)
    if normalized["queries"] != 2 * images:
        raise ValueError("Validation must contain exactly two side queries per image")
    for side in SIDES:
        positives = _integer(normalized[f"{side}/positive_count"],
                             f"{side}/positive_count", minimum=1)
        absent = normalized[f"{side}/absent_count"]
        if positives + absent != images:
            raise ValueError(f"{side} positive + absent counts must cover every image")
        for key, upper in (("false_negative_count", positives),
                           ("false_positive_count", absent)):
            full_key = f"{side}/{key}"
            count = _integer(metrics[full_key], full_key)
            if count > upper:
                raise ValueError(f"{full_key} exceeds its denominator")
            normalized[full_key] = count
        score = _finite(normalized[f"{side}/miss_zero_dice"], f"{side}/miss_zero_dice")
        if not 0. <= score <= 1.:
            raise ValueError(f"{side}/miss_zero_dice must be in [0, 1]")
        normalized[f"{side}/miss_zero_dice"] = score
    if normalized["targets"] != sum(normalized[f"{side}/positive_count"] for side in SIDES):
        raise ValueError("Validation target count must equal the two positive counts")
    return normalized


def selection_metric(metrics):
    """Return an equal-side macro mean, never a positive-count-weighted mean."""
    values = _validation_metrics(metrics)
    return sum(values[f"{side}/miss_zero_dice"] for side in SIDES) / 2.


def new_selection_state(validation_sha256, *, patience=3, min_delta=0.001):
    """Create an empty selector bound to fixed DexYCB validation annotations."""
    return {
        "schema": SCHEMA,
        "metric_name": METRIC_NAME,
        "mode": "max",
        "validation_scope": SCOPE,
        "validation_sha256": _digest(validation_sha256),
        "policy": _policy(patience, min_delta),
        "best_metric": None,
        "best_step": None,
        "best_epoch": None,
        "early_stop_reference_metric": None,
        "bad_epochs": 0,
        "last_validation_step": None,
        "last_validation_epoch": None,
        "stopped": False,
        "history": [],
    }


def _advance(state, metrics, *, step, epoch, validation_sha256, scope):
    # state is a newly created/replayed private copy, so input state is untouched.
    if scope != SCOPE:
        raise ValueError("Selection is restricted to fixed dexycb_val, never RealSense/test")
    if _digest(validation_sha256) != state["validation_sha256"]:
        raise ValueError("Validation annotations SHA256 changed")
    if state["stopped"]:
        raise ValueError("Selection is terminal after patience is exhausted")
    step = _integer(step, "step")
    epoch = _integer(epoch, "epoch")
    if state["last_validation_step"] is not None and step <= state["last_validation_step"]:
        raise ValueError("Validation step must strictly increase; duplicates are rejected")
    if state["last_validation_epoch"] is not None and epoch <= state["last_validation_epoch"]:
        raise ValueError("Validation epoch must strictly increase; one decision per epoch")
    values = _validation_metrics(metrics)
    if state["history"]:
        original = state["history"][0]["metrics"]
        if any(values[key] != original[key] for key in _FIXED_COUNTS):
            raise ValueError("Fixed validation population/side counts changed")
    metric = sum(values[f"{side}/miss_zero_dice"] for side in SIDES) / 2.
    baseline = state["best_metric"] is None
    is_best = baseline or metric > state["best_metric"]
    significant = (baseline or metric >
                   state["early_stop_reference_metric"] + state["policy"]["min_delta"])
    if is_best:
        state.update(best_metric=metric, best_step=step, best_epoch=epoch)
    if significant:
        state["early_stop_reference_metric"] = metric
        state["bad_epochs"] = 0
    else:
        state["bad_epochs"] += 1
    state["stopped"] = state["bad_epochs"] >= state["policy"]["patience"]
    state.update(last_validation_step=step, last_validation_epoch=epoch)
    decision = {
        "metric": metric,
        "is_baseline": baseline,
        "is_best": is_best,
        "significant_improvement": significant,
        "best_metric": state["best_metric"],
        "best_step": state["best_step"],
        "bad_epochs": state["bad_epochs"],
        "should_stop": state["stopped"],
        "stop_reason": "patience_exhausted" if state["stopped"] else None,
    }
    # Separate dictionaries prevent later mutation of the returned decision
    # from changing the checkpoint history.
    state["history"].append({"step": step, "epoch": epoch, "scope": scope,
                             "validation_sha256": validation_sha256,
                             "metrics": values, "decision": dict(decision)})
    return state, decision


def _json(value):
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Selection state must contain only finite JSON-native data") from exc


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate selection checkpoint JSON key: {key}")
        result[key] = value
    return result


def _validate_state(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("Selection state must be a mapping")
    try:
        policy = payload["policy"]
        if not isinstance(policy, Mapping) or set(policy) != {"patience", "min_delta"}:
            raise ValueError("Malformed selection policy")
        rebuilt = new_selection_state(payload["validation_sha256"], **policy)
        if set(payload) != set(rebuilt) or payload["schema"] != SCHEMA:
            raise ValueError("Unknown selection schema or missing/extra state fields")
        if not isinstance(payload["history"], list):
            raise ValueError("Selection history must be a list")
        for record in payload["history"]:
            if not isinstance(record, Mapping):
                raise ValueError("Malformed selection history entry")
            rebuilt, _ = _advance(rebuilt, record["metrics"], step=record["step"],
                                  epoch=record["epoch"], scope=record["scope"],
                                  validation_sha256=record["validation_sha256"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing or malformed selection state/history field") from exc
    if _json(payload) != _json(rebuilt):
        raise ValueError("Selection state or history differs from deterministic replay")
    return rebuilt


def update_selection(state, metrics, *, step, epoch, validation_sha256, scope=SCOPE):
    """Pure transition returning (new_state, decision), without modifying inputs.

    Both step and epoch are cumulative across resumes. A repeated validation is
    an error even if its metrics match. The initial observation establishes a
    best baseline and consumes no patience, including when its step is nonzero.
    """
    copied = _validate_state(state)
    return _advance(copied, metrics, step=step, epoch=epoch,
                    validation_sha256=validation_sha256, scope=scope)


def serialize_selection_state(state):
    """Validate and return deterministic JSON; state itself also fits torch.save."""
    return _json(_validate_state(state))


def restore_selection_state(payload, *, validation_sha256, patience=3, min_delta=0.001):
    """Restore a JSON string or checkpoint mapping against the active contract.

    Supplying the active CLI policy is important: changing patience/min_delta on
    resume is rejected rather than silently rewriting historical decisions.
    Missing state in a legacy checkpoint is an error, not an implicit reset.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload, object_pairs_hook=_unique_json_object)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid selection checkpoint JSON") from exc
    restored = _validate_state(payload)
    if restored["validation_sha256"] != _digest(validation_sha256):
        raise ValueError("Selection checkpoint validation annotations SHA256 differs")
    if restored["policy"] != _policy(patience, min_delta):
        raise ValueError("Selection checkpoint policy differs from the active policy")
    return restored
