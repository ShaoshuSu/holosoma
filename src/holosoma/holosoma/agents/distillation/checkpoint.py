"""Read the policy structure a saved distillation checkpoint was built with."""

from __future__ import annotations

from typing import Any


def infer_num_teachers_from_checkpoint(checkpoint: dict[str, Any]) -> int | None:
    """Read the teacher-slot count from a full distillation checkpoint."""
    explicit_count = checkpoint.get("num_teachers")
    if explicit_count is not None:
        if type(explicit_count) is not int or explicit_count < 1:
            raise ValueError(f"Invalid num_teachers checkpoint metadata: {explicit_count!r}")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        return explicit_count

    legacy_single_teacher = any(key.startswith("teacher.") for key in state_dict)
    indices: set[int] = set()
    for key in state_dict:
        parts = key.split(".", 2)
        if len(parts) >= 2 and parts[0] == "teachers" and parts[1].isdigit():
            indices.add(int(parts[1]))

    if legacy_single_teacher and indices:
        raise ValueError("Checkpoint mixes legacy 'teacher.*' and current 'teachers.*' keys.")
    if legacy_single_teacher:
        inferred_count = 1
    elif indices:
        expected = set(range(max(indices) + 1))
        if indices != expected:
            raise ValueError(f"Checkpoint has non-contiguous teacher slots: {sorted(indices)}.")
        inferred_count = len(indices)
    else:
        return explicit_count
    if explicit_count is not None and explicit_count != inferred_count:
        raise ValueError(
            f"Checkpoint num_teachers metadata says {explicit_count}, but its state dict "
            f"contains {inferred_count} teacher slots."
        )
    return inferred_count
