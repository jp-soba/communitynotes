import argparse
import logging
import os
import sys

from . import constants as c
from .enums import scorers_from_csv
from .pandas_utils import patch_pandas
from .process_data import (
  LocalDataLoader,
  tsv_reader,
  write_parquet_local,
  write_prescoring_output,
  write_tsv_local,
)
from .run_scoring import run_final_phase, run_prescoring_phase, run_scoring

import pandas as pd


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
    "--phase",
    default="all",
    choices=["all", "prescoring", "final"],
    dest="phase",
    help=(
      "Which scoring phase to run. 'all' (default) runs prescoring and final scoring in one "
      "process (original behavior). 'prescoring' runs only prescoring and writes its output to "
      "disk; 'final' loads that prescoring output and runs final + contributor scoring. Splitting "
      "into two processes lowers peak RAM, since the prescoring working set is freed on process "
      "exit before final scoring begins."
    ),
  )
  parser.add_argument(
    "--prescoring-note-model-output",
    default=None,
    dest="prescoring_note_model_output",
    help="path for prescoring note model output (defaults to <outdir>/prescoring_note_model_output.tsv)",
  )
  parser.add_argument(
    "--prescoring-rater-model-output",
    default=None,
    dest="prescoring_rater_model_output",
    help="path for prescoring rater model output (defaults to <outdir>/prescoring_rater_model_output.tsv)",
  )
  parser.add_argument(
    "--prescoring-note-topic-classifier",
    default=None,
    dest="prescoring_note_topic_classifier",
    help="path for prescoring note topic classifier (defaults to <outdir>/prescoring_note_topic_classifier.joblib)",
  )
  parser.add_argument(
    "--prescoring-pflip-classifier",
    default=None,
    dest="prescoring_pflip_classifier",
    help="path for prescoring pflip classifier (defaults to <outdir>/prescoring_pflip_classifier.pickle)",
  )
  parser.add_argument(
    "--prescoring-meta-output",
    default=None,
    dest="prescoring_meta_output",
    help="path for prescoring meta output (defaults to <outdir>/prescoring_meta_output.joblib)",
  )
  parser.add_argument(
    "--prescoring-empirical-totals",
    default=None,
    dest="prescoring_empirical_totals",
    help="path for prescoring empirical totals (defaults to <outdir>/prescoring_empirical_totals.parquet)",
  )
  return parser.parse_args()


def _prescoring_artifact_paths(args):
  """Resolve paths for the on-disk prescoring artifacts, defaulting to files under outdir."""

  def resolve(value, filename):
    return value if value is not None else os.path.join(args.outdir, filename)

  return {
    "note_model_output": resolve(
      args.prescoring_note_model_output, "prescoring_note_model_output.tsv"
    ),
    "rater_model_output": resolve(
      args.prescoring_rater_model_output, "prescoring_rater_model_output.tsv"
    ),
    "note_topic_classifier": resolve(
      args.prescoring_note_topic_classifier, "prescoring_note_topic_classifier.joblib"
    ),
    "pflip_classifier": resolve(
      args.prescoring_pflip_classifier, "prescoring_pflip_classifier.pickle"
    ),
    "meta_output": resolve(args.prescoring_meta_output, "prescoring_meta_output.joblib"),
    "empirical_totals": resolve(
      args.prescoring_empirical_totals, "prescoring_empirical_totals.parquet"
    ),
  }


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

  # Invoke scoring and user contribution algorithms. The phase determines whether prescoring and
  # final scoring run together (default) or as separate processes to lower peak RAM.
  if args.phase == "prescoring":
    paths = _prescoring_artifact_paths(args)
    (
      prescoringNoteModelOutput,
      prescoringRaterModelOutput,
      prescoringNoteTopicClassifier,
      prescoringPflipClassifier,
      prescoringMetaOutput,
      _prescoringScoredNotes,
      empiricalTotals,
    ) = run_prescoring_phase(
      args,
      notes,
      ratings,
      statusHistory,
      userEnrollment,
      seed=args.seed,
      enabledScorers=args.scorers,
      runParallel=args.parallel,
      dataLoader=dataLoader if args.parallel == True else None,
      cutoffTimestampMillis=args.cutoffTimestampMillis,
      excludeRatingsAfterANoteGotFirstStatusPlusNHours=args.excludeRatingsAfterANoteGotFirstStatusPlusNHours,
      daysInPastToApplyPostFirstStatusFiltering=args.daysInPastToApplyPostFirstStatusFiltering,
      filterPrescoringInputToSimulateDelayInHours=args.prescoring_delay_hours,
      previousRatingCutoffTimestampMillis=args.previous_rating_cutoff_millis,
      **extraScoringArgs,
    )
    write_prescoring_output(
      prescoringNoteModelOutput,
      prescoringRaterModelOutput,
      prescoringNoteTopicClassifier,
      prescoringPflipClassifier,
      prescoringMetaOutput,
      None,
      paths["note_model_output"],
      paths["rater_model_output"],
      paths["note_topic_classifier"],
      paths["pflip_classifier"],
      paths["meta_output"],
      None,
      headers=args.headers,
    )
    if empiricalTotals is not None:
      write_parquet_local(empiricalTotals, paths["empirical_totals"])
    logger.info("Prescoring phase complete; prescoring output written to disk.")
    return

  if args.phase == "final":
    paths = _prescoring_artifact_paths(args)
    prescoringLoader = LocalDataLoader(
      args.notes,
      args.ratings,
      args.status,
      args.enrollment,
      args.headers,
      prescoringNoteModelOutputPath=paths["note_model_output"],
      prescoringRaterModelOutputPath=paths["rater_model_output"],
      prescoringNoteTopicClassifierPath=paths["note_topic_classifier"],
      prescoringPflipClassifierPath=paths["pflip_classifier"],
      prescoringMetaOutputPath=paths["meta_output"],
    )
    (
      prescoringNoteModelOutput,
      prescoringRaterModelOutput,
      prescoringNoteTopicClassifier,
      prescoringPflipClassifier,
      prescoringMetaOutput,
    ) = prescoringLoader.get_prescoring_model_output()
    empiricalTotals = (
      pd.read_parquet(paths["empirical_totals"])
      if os.path.exists(paths["empirical_totals"])
      else None
    )
    scoredNotes, helpfulnessScores, newStatus, auxNoteInfo = run_final_phase(
      args,
      notes,
      ratings,
      statusHistory,
      userEnrollment,
      prescoringNoteModelOutput,
      prescoringRaterModelOutput,
      prescoringNoteTopicClassifier,
      prescoringPflipClassifier,
      prescoringMetaOutput,
      empiricalTotals=empiricalTotals,
      seed=args.seed,
      pseudoraters=args.pseudoraters,
      enabledScorers=args.scorers,
      strictColumns=args.strict_columns,
      runParallel=args.parallel,
      dataLoader=dataLoader if args.parallel == True else None,
      checkFlips=args.check_flips,
      previousScoredNotes=previousScoredNotes,
      previousAuxiliaryNoteInfo=previousAuxiliaryNoteInfo,
      previousRatingCutoffTimestampMillis=args.previous_rating_cutoff_millis,
      cutoffTimestampMillis=args.cutoffTimestampMillis,
      excludeRatingsAfterANoteGotFirstStatusPlusNHours=args.excludeRatingsAfterANoteGotFirstStatusPlusNHours,
      daysInPastToApplyPostFirstStatusFiltering=args.daysInPastToApplyPostFirstStatusFiltering,
      filterPrescoringInputToSimulateDelayInHours=args.prescoring_delay_hours,
      **extraScoringArgs,
    )
  else:
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
      **extraScoringArgs,
    )

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
