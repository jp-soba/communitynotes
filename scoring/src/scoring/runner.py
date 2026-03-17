import argparse
import logging
import os
import sys

import joblib
import pandas as pd

from . import constants as c
from .enums import scorers_from_csv
from .id_mapping import (
  apply_mapping_to_all,
  build_participant_id_mapping,
  reverse_mapping_on_outputs,
)
from .pandas_utils import patch_pandas
from .process_data import (
  LocalDataLoader,
  filter_input_data_for_testing,
  tsv_reader,
  write_parquet_local,
  write_prescoring_output,
  write_tsv_local,
)
from .run_scoring import (
  run_contributor_scoring,
  run_final_note_scoring,
  run_scoring,
)


logger = logging.getLogger("birdwatch.runner")
logger.setLevel(logging.INFO)


def parse_args():
  parser = argparse.ArgumentParser("Community Notes Scoring")
  parser.add_argument(
    "--check-flips",
    dest="check_flips",
    help="Validate that note statuses align with prior runs (disable for testing)",
    action="store_true",
  )
  parser.add_argument(
    "--nocheck-flips",
    help="Disable validation that note statuses align with prior runs (use for testing)",
    action="store_false",
    dest="check_flips",
  )
  parser.set_defaults(check_flips=False)
  parser.add_argument(
    "--enforce-types",
    dest="enforce_types",
    help="Raise errors when types in Pandas operations do not meet expectations.",
    action="store_true",
  )
  parser.add_argument(
    "--noenforce-types",
    dest="enforce_types",
    help="Log to stderr when types in Pandas operations do not meet expectations.",
    action="store_false",
  )
  parser.set_defaults(enforce_types=False)
  parser.add_argument(
    "-e", "--enrollment", default=c.enrollmentInputPath, help="note enrollment dataset"
  )
  parser.add_argument(
    "--epoch-millis",
    default=None,
    type=float,
    dest="epoch_millis",
    help="timestamp in milliseconds since epoch to treat as now",
  )
  parser.add_argument(
    "--headers",
    dest="headers",
    help="First row of input files should be a header",
    action="store_true",
  )
  parser.add_argument(
    "--noheaders",
    dest="headers",
    help="First row of input files should be data.  There should be no headers.",
    action="store_false",
  )
  parser.set_defaults(headers=True)
  parser.add_argument("-n", "--notes", default=c.notesInputPath, help="note dataset")
  parser.add_argument(
    "--previous-scored-notes", default=None, help="previous scored notes dataset path"
  )
  parser.add_argument(
    "--previous-aux-note-info", default=None, help="previous aux note info dataset path"
  )
  parser.add_argument(
    "--previous-rating-cutoff-millis",
    default=None,
    type=int,
    help="previous rating cutoff millis",
  )
  parser.add_argument("-o", "--outdir", default=".", help="directory for output files")
  parser.add_argument(
    "--pseudoraters",
    dest="pseudoraters",
    help="Include calculation of pseudorater intervals",
    action="store_true",
  )
  parser.add_argument(
    "--nopseudoraters",
    dest="pseudoraters",
    help="Exclude calculation of pseudorater intervals (faster)",
    action="store_false",
  )
  parser.set_defaults(pseudoraters=True)
  parser.add_argument("-r", "--ratings", default=c.ratingsInputPath, help="rating dataset")
  parser.add_argument(
    "--scorers",
    default=None,
    type=scorers_from_csv,
    help="CSV list of scorers to enable.",
  )
  parser.add_argument(
    "--seed", default=None, type=int, help="set to an int to seed matrix factorization"
  )
  parser.add_argument(
    "-s",
    "--status",
    default=c.noteStatusHistoryInputPath,
    help="note status history dataset",
  )
  parser.add_argument(
    "--strict-columns",
    dest="strict_columns",
    help="Explicitly select columns and require that expected columns are present.",
    action="store_true",
  )
  parser.add_argument(
    "--nostrict-columns",
    help="Disable validation of expected columns and allow unexpected columns.",
    action="store_false",
    dest="strict_columns",
  )
  parser.set_defaults(strict_columns=True)
  parser.add_argument(
    "--parallel",
    help="Enable parallel run of algorithm.",
    action="store_true",
    dest="parallel",
  )
  parser.set_defaults(parallel=False)

  parser.add_argument(
    "--no-parquet",
    help="Disable writing parquet files.",
    default=False,
    action="store_true",
    dest="no_parquet",
  )

  parser.add_argument(
    "--cutoff-timestamp-millis",
    default=None,
    type=int,
    dest="cutoffTimestampMillis",
    help="filter notes and ratings created after this time.",
  )
  parser.add_argument(
    "--exclude-ratings-after-a-note-got-first-status-plus-n-hours",
    default=None,
    type=int,
    dest="excludeRatingsAfterANoteGotFirstStatusPlusNHours",
    help="Exclude ratings after a note got first status plus n hours",
  )
  parser.add_argument(
    "--days-in-past-to-apply-post-first-status-filtering",
    default=14,
    type=int,
    dest="daysInPastToApplyPostFirstStatusFiltering",
    help="Days in past to apply post first status filtering",
  )
  parser.add_argument(
    "--prescoring-delay-hours",
    default=None,
    type=int,
    dest="prescoring_delay_hours",
    help="Filter prescoring input to simulate delay in hours",
  )
  parser.add_argument(
    "--sample-ratings",
    default=0.0,
    type=float,
    dest="sample_ratings",
    help="Set to sample ratings at random.",
  )
  parser.add_argument(
    "--prescoring-dir",
    default=None,
    dest="prescoring_dir",
    help="Directory to save/load prescoring checkpoint. When set, prescoring results are saved after completion.",
  )
  parser.add_argument(
    "--resume-from-prescoring",
    default=False,
    action="store_true",
    dest="resume_from_prescoring",
    help="Resume from a saved prescoring checkpoint (requires --prescoring-dir).",
  )
  return parser.parse_args()


@patch_pandas
def _run_scorer(
  args=None,
  dataLoader=None,
  extraScoringArgs={},
):
  logger.info("beginning scorer execution")
  assert args is not None, "args must be available"
  if args.epoch_millis:
    c.epochMillis = args.epoch_millis
    c.useCurrentTimeInsteadOfEpochMillisForNoteStatusHistory = False

  # Load input dataframes.
  if dataLoader is None:
    dataLoader = LocalDataLoader(
      args.notes,
      args.ratings,
      args.status,
      args.enrollment,
      args.headers,
    )
  notes, ratings, statusHistory, userEnrollment = dataLoader.get_data()

  # Map string participant IDs to sequential int64 values for memory efficiency.
  # This is a no-op when IDs are already numeric (i.e. production data).
  id_mapping = build_participant_id_mapping([
    (notes, [c.noteAuthorParticipantIdKey]),
    (ratings, [c.raterParticipantIdKey]),
    (statusHistory, [c.noteAuthorParticipantIdKey]),
    (userEnrollment, [c.participantIdKey]),
  ])
  if id_mapping is not None:
    apply_mapping_to_all(id_mapping, notes, ratings, statusHistory, userEnrollment)
    reverse_map = {v: k for k, v in id_mapping.items()}
    del id_mapping
  else:
    reverse_map = None

  if args.previous_scored_notes is not None:
    previousScoredNotes = tsv_reader(
      args.previous_scored_notes,
      c.noteModelOutputTSVTypeMapping,
      c.noteModelOutputTSVColumns,
      header=False,
      convertNAToNone=False,
    )
    assert (
      args.previous_aux_note_info is not None
    ), "previous_aux_note_info must be available if previous_scored_notes is available"
    previousAuxiliaryNoteInfo = tsv_reader(
      args.previous_aux_note_info,
      c.auxiliaryScoredNotesTSVTypeMapping,
      c.auxiliaryScoredNotesTSVColumns,
      header=False,
      convertNAToNone=False,
    )
  else:
    previousScoredNotes = None
    previousAuxiliaryNoteInfo = None

  # Sample ratings to decrease runtime
  if args.sample_ratings:
    origSize = len(ratings)
    ratings = ratings.sample(frac=args.sample_ratings)
    logger.info(f"ratings reduced from {origSize} to {len(ratings)}")

  # Resume from prescoring checkpoint if requested.
  if getattr(args, "resume_from_prescoring", False):
    assert getattr(args, "prescoring_dir", None) is not None, (
      "--prescoring-dir is required when using --resume-from-prescoring"
    )
    scoredNotes, helpfulnessScores, newStatus, auxNoteInfo = _run_from_prescoring_checkpoint(
      args, notes, ratings, statusHistory, userEnrollment,
      reverse_map=reverse_map,
      previousScoredNotes=previousScoredNotes,
      previousAuxiliaryNoteInfo=previousAuxiliaryNoteInfo,
    )
  else:
    # Build prescoring save callback if --prescoring-dir is set.
    prescoringCallback = None
    prescoringDir = getattr(args, "prescoring_dir", None)
    if prescoringDir is not None:
      os.makedirs(prescoringDir, exist_ok=True)
      def prescoringCallback(
        noteModel, raterModel, topicClassifier, pflipClassifier,
        metaOutput, scoredNotesOutput, empiricalTotals,
      ):
        _save_prescoring_checkpoint(
          prescoringDir, noteModel, raterModel, topicClassifier,
          pflipClassifier, metaOutput, scoredNotesOutput, empiricalTotals,
        )

    # Invoke scoring and user contribution algorithms.
    scoredNotes, helpfulnessScores, newStatus, auxNoteInfo = run_scoring(
      args,
      notes,
      ratings,
      statusHistory,
      userEnrollment,
      seed=args.seed,
      pseudoraters=args.pseudoraters,
      enabledScorers=args.scorers,
      strictColumns=args.strict_columns,
      runParallel=args.parallel,
      dataLoader=dataLoader if args.parallel == True else None,
      cutoffTimestampMillis=args.cutoffTimestampMillis,
      excludeRatingsAfterANoteGotFirstStatusPlusNHours=args.excludeRatingsAfterANoteGotFirstStatusPlusNHours,
      daysInPastToApplyPostFirstStatusFiltering=args.daysInPastToApplyPostFirstStatusFiltering,
      filterPrescoringInputToSimulateDelayInHours=args.prescoring_delay_hours,
      checkFlips=args.check_flips,
      previousScoredNotes=previousScoredNotes,
      previousAuxiliaryNoteInfo=previousAuxiliaryNoteInfo,
      previousRatingCutoffTimestampMillis=args.previous_rating_cutoff_millis,
      writePrescoringScoringOutputCallback=prescoringCallback,
      **extraScoringArgs,
    )

  # Restore original string participant IDs before writing output.
  if reverse_map is not None:
    reverse_mapping_on_outputs(reverse_map, scoredNotes, helpfulnessScores, newStatus, auxNoteInfo)
    del reverse_map

  # Write outputs to local disk.
  write_tsv_local(scoredNotes, os.path.join(args.outdir, "scored_notes.tsv"))
  write_tsv_local(helpfulnessScores, os.path.join(args.outdir, "helpfulness_scores.tsv"))
  write_tsv_local(newStatus, os.path.join(args.outdir, "note_status_history.tsv"))
  write_tsv_local(auxNoteInfo, os.path.join(args.outdir, "aux_note_info.tsv"))

  if not args.no_parquet:
    write_parquet_local(scoredNotes, os.path.join(args.outdir, "scored_notes.parquet"))
    write_parquet_local(helpfulnessScores, os.path.join(args.outdir, "helpfulness_scores.parquet"))
    write_parquet_local(newStatus, os.path.join(args.outdir, "note_status_history.parquet"))
    write_parquet_local(auxNoteInfo, os.path.join(args.outdir, "aux_note_info.parquet"))


def _save_prescoring_checkpoint(
  prescoringDir,
  prescoringNoteModelOutput,
  prescoringRaterModelOutput,
  noteTopicClassifier,
  pflipClassifier,
  prescoringMetaOutput,
  prescoringScoredNotes,
  empiricalTotals,
):
  """Save prescoring results to disk as a checkpoint."""
  logger.info(f"Saving prescoring checkpoint to {prescoringDir}")
  write_prescoring_output(
    prescoringNoteModelOutput,
    prescoringRaterModelOutput,
    noteTopicClassifier,
    pflipClassifier,
    prescoringMetaOutput,
    prescoringScoredNotes,
    noteModelOutputPath=os.path.join(prescoringDir, "prescoring_note_model_output.tsv"),
    raterModelOutputPath=os.path.join(prescoringDir, "prescoring_rater_model_output.tsv"),
    noteTopicClassifierPath=os.path.join(prescoringDir, "note_topic_classifier.joblib"),
    pflipClassifierPath=os.path.join(prescoringDir, "pflip_classifier.bin"),
    prescoringMetaOutputPath=os.path.join(prescoringDir, "prescoring_meta_output.joblib"),
    prescoringScoredNotesOutputPath=os.path.join(prescoringDir, "prescoring_scored_notes.tsv"),
  )
  if empiricalTotals is not None:
    write_tsv_local(empiricalTotals, os.path.join(prescoringDir, "empirical_totals.tsv"))
  logger.info("Prescoring checkpoint saved.")


def _load_prescoring_checkpoint(prescoringDir):
  """Load prescoring results from a checkpoint directory."""
  logger.info(f"Loading prescoring checkpoint from {prescoringDir}")
  prescoringNoteModelOutput = tsv_reader(
    os.path.join(prescoringDir, "prescoring_note_model_output.tsv"),
    c.prescoringNoteModelOutputTSVTypeMapping,
    c.prescoringNoteModelOutputTSVColumns,
    header=True,
    convertNAToNone=False,
  )
  prescoringRaterModelOutput = tsv_reader(
    os.path.join(prescoringDir, "prescoring_rater_model_output.tsv"),
    c.prescoringRaterModelOutputTSVTypeMapping,
    c.prescoringRaterModelOutputTSVColumns,
    header=True,
    convertNAToNone=False,
  )
  noteTopicClassifier = joblib.load(
    os.path.join(prescoringDir, "note_topic_classifier.joblib")
  )
  pflipClassifier = joblib.load(os.path.join(prescoringDir, "pflip_classifier.bin"))
  prescoringMetaOutput = joblib.load(
    os.path.join(prescoringDir, "prescoring_meta_output.joblib")
  )
  scoredNotesPath = os.path.join(prescoringDir, "prescoring_scored_notes.tsv")
  if os.path.exists(scoredNotesPath):
    prescoringScoredNotes = pd.read_csv(scoredNotesPath, sep="\t")
  else:
    prescoringScoredNotes = None
  empiricalTotalsPath = os.path.join(prescoringDir, "empirical_totals.tsv")
  if os.path.exists(empiricalTotalsPath):
    empiricalTotals = pd.read_csv(empiricalTotalsPath, sep="\t")
  else:
    empiricalTotals = None
  logger.info("Prescoring checkpoint loaded.")
  return (
    prescoringNoteModelOutput,
    prescoringRaterModelOutput,
    noteTopicClassifier,
    pflipClassifier,
    prescoringMetaOutput,
    prescoringScoredNotes,
    empiricalTotals,
  )


def _run_from_prescoring_checkpoint(
  args, notes, ratings, noteStatusHistory, userEnrollment,
  reverse_map=None, previousScoredNotes=None, previousAuxiliaryNoteInfo=None,
):
  """Resume scoring from a prescoring checkpoint, skipping prescoring entirely."""
  (
    prescoringNoteModelOutput,
    prescoringRaterModelOutput,
    noteTopicClassifier,
    pflipClassifier,
    prescoringMetaOutput,
    prescoringScoredNotes,
    empiricalTotals,
  ) = _load_prescoring_checkpoint(args.prescoring_dir)

  # If ID mapping was applied, prescoring checkpoint was saved with mapped int IDs
  # serialized as strings. Re-apply the mapping so types match the input DataFrames.
  if reverse_map is not None:
    # Build forward mapping from the reverse map for re-mapping loaded checkpoint data.
    forward_map = {v: k for k, v in reverse_map.items()}
    from .id_mapping import apply_mapping
    apply_mapping(prescoringRaterModelOutput, forward_map, [c.raterParticipantIdKey])

  # Apply the same input filtering that run_scoring() applies.
  notes, ratings, _, _ = filter_input_data_for_testing(
    notes,
    ratings,
    noteStatusHistory,
    getattr(args, "cutoffTimestampMillis", None),
    getattr(args, "excludeRatingsAfterANoteGotFirstStatusPlusNHours", None),
    getattr(args, "daysInPastToApplyPostFirstStatusFiltering", 14),
    getattr(args, "prescoring_delay_hours", None),
  )

  logger.info("Starting final scoring (resumed from prescoring checkpoint)")
  scoredNotes, newNoteStatusHistory, auxiliaryNoteInfo, _ = run_final_note_scoring(
    args,
    notes=notes,
    ratings=ratings,
    noteStatusHistory=noteStatusHistory,
    userEnrollment=userEnrollment,
    seed=getattr(args, "seed", None),
    pseudoraters=getattr(args, "pseudoraters", True),
    enabledScorers=getattr(args, "scorers", None),
    strictColumns=getattr(args, "strict_columns", True),
    runParallel=getattr(args, "parallel", False),
    dataLoader=None,
    prescoringNoteModelOutput=prescoringNoteModelOutput,
    prescoringRaterModelOutput=prescoringRaterModelOutput,
    noteTopicClassifier=noteTopicClassifier,
    pflipClassifier=pflipClassifier,
    prescoringMetaOutput=prescoringMetaOutput,
    previousScoredNotes=previousScoredNotes,
    previousAuxiliaryNoteInfo=previousAuxiliaryNoteInfo,
    previousRatingCutoffTimestampMillis=getattr(args, "previous_rating_cutoff_millis", 0),
    empiricalTotals=empiricalTotals,
  )

  logger.info("Starting contributor scoring (resumed from prescoring checkpoint)")
  helpfulnessScores = run_contributor_scoring(
    ratings=ratings,
    scoredNotes=scoredNotes,
    auxiliaryNoteInfo=auxiliaryNoteInfo,
    prescoringRaterModelOutput=prescoringRaterModelOutput,
    noteStatusHistory=newNoteStatusHistory,
    userEnrollment=userEnrollment,
    strictColumns=getattr(args, "strict_columns", True),
  )

  return scoredNotes, helpfulnessScores, newNoteStatusHistory, auxiliaryNoteInfo


def main(
  args=None,
  dataLoader=None,
  extraScoringArgs={},
):
  if args is None:
    args = parse_args()
  logger.info(f"scorer python version: {sys.version}")
  logger.info(f"scorer pandas version: {pd.__version__}")
  # patch_pandas requires that args are available (which matches the production binary) so
  # we first parse the arguments then invoke the decorated _run_scorer.
  return _run_scorer(args=args, dataLoader=dataLoader, extraScoringArgs=extraScoringArgs)


if __name__ == "__main__":
  main()
