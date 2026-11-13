"""Shared default path for the supplied shared_aux_transfer_v1 teacher."""
from pathlib import Path


DEFAULT_TEACHER_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "pretrained"
    / "teacher_canya_cpad.pt"
)
