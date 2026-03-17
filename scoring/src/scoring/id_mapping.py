"""Utility for mapping string participant IDs to sequential int64 values.

Public Community Notes data uses 64-character hex hashes as participant IDs,
which are stored as Python objects in pandas and consume ~121 bytes per cell.
Mapping these to sequential int64 values (8 bytes per cell) dramatically
reduces memory usage and speeds up groupby/merge operations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import constants as c

logger = logging.getLogger("birdwatch.id_mapping")

# All participant ID column names that share the same entity namespace.
_PARTICIPANT_ID_COLUMNS = [
  c.participantIdKey,
  c.noteAuthorParticipantIdKey,
  c.raterParticipantIdKey,
]


def _is_string_ids(df: pd.DataFrame, col: str) -> bool:
  """Check if a column contains string IDs (not already numeric)."""
  if col not in df.columns:
    return False
  return df[col].dtype == object


def build_participant_id_mapping(
  dataframes: List[Tuple[pd.DataFrame, List[str]]],
) -> Optional[Dict[str, int]]:
  """Build a mapping from unique string participant IDs to sequential integers.

  Args:
    dataframes: List of (DataFrame, [column_names]) tuples. Each column_name
      should be a participant ID column present in the DataFrame.

  Returns:
    Dict mapping string IDs to int64 values, or None if IDs are already numeric.
  """
  # Collect all unique participant IDs across all DataFrames and columns.
  all_ids = set()
  has_string_ids = False

  for df, columns in dataframes:
    for col in columns:
      if col in df.columns and _is_string_ids(df, col):
        has_string_ids = True
        all_ids.update(df[col].dropna().unique())

  if not has_string_ids:
    logger.info("Participant IDs are already numeric. Skipping ID mapping.")
    return None

  # Sort for deterministic mapping, then assign sequential integers starting from 1.
  # 0 is reserved so that any accidental default/missing values are distinguishable.
  sorted_ids = sorted(all_ids)
  mapping = {id_str: idx + 1 for idx, id_str in enumerate(sorted_ids)}
  logger.info(f"Built participant ID mapping: {len(mapping)} unique IDs.")
  return mapping


def apply_mapping(
  df: pd.DataFrame,
  mapping: Dict[str, int],
  columns: List[str],
) -> pd.DataFrame:
  """Replace string participant IDs with mapped int64 values in-place.

  Args:
    df: DataFrame to modify.
    mapping: String-to-int mapping from build_participant_id_mapping.
    columns: List of column names to convert.

  Returns:
    The same DataFrame (modified in-place).
  """
  for col in columns:
    if col in df.columns and _is_string_ids(df, col):
      df[col] = df[col].map(mapping).astype(np.int64)
  return df


def reverse_mapping(
  df: pd.DataFrame,
  reverse_map: Dict[int, str],
  columns: List[str],
) -> pd.DataFrame:
  """Restore original string participant IDs from int64 values in-place.

  Args:
    df: DataFrame to modify.
    reverse_map: Int-to-string mapping (inverse of build_participant_id_mapping result).
    columns: List of column names to convert.

  Returns:
    The same DataFrame (modified in-place).
  """
  for col in columns:
    if col in df.columns and df[col].dtype in (np.int64, np.dtype("int64")):
      df[col] = df[col].map(reverse_map)
  return df


def apply_mapping_to_all(
  mapping: Dict[str, int],
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
) -> None:
  """Apply ID mapping to all input DataFrames."""
  apply_mapping(notes, mapping, [c.noteAuthorParticipantIdKey])
  apply_mapping(ratings, mapping, [c.raterParticipantIdKey])
  apply_mapping(noteStatusHistory, mapping, [c.noteAuthorParticipantIdKey])
  apply_mapping(userEnrollment, mapping, [c.participantIdKey])


def reverse_mapping_on_outputs(
  reverse_map: Dict[int, str],
  scoredNotes: pd.DataFrame,
  helpfulnessScores: pd.DataFrame,
  newStatus: pd.DataFrame,
  auxNoteInfo: pd.DataFrame,
) -> None:
  """Reverse ID mapping on all output DataFrames."""
  reverse_mapping(scoredNotes, reverse_map, [c.noteAuthorParticipantIdKey])
  reverse_mapping(helpfulnessScores, reverse_map, [c.raterParticipantIdKey])
  reverse_mapping(newStatus, reverse_map, [c.noteAuthorParticipantIdKey])
  reverse_mapping(auxNoteInfo, reverse_map, [c.noteAuthorParticipantIdKey])
