# ML pipeline — dataset construction, training, evaluation, and deployment.
from morningalpha import cli_context
import rich_click as click
from morningalpha.ml.dataset import dataset as ml_dataset
from morningalpha.ml.train import train as ml_train_lgbm, wfcv as ml_wfcv
from morningalpha.fundamentals import fundamentals_cmd as ml_fundamentals
from morningalpha.ml.backtest import backtest as ml_backtest
from morningalpha.ml.score import score as ml_score
from morningalpha.ml.backfill import backfill as ml_backfill, seed_calibration_daily as ml_seed_calib
from morningalpha.ml.train_lstm import train_lstm as ml_train_lstm


@click.group(context_settings=cli_context)
def ml():
    """ML pipeline commands — dataset, training, evaluation, and deployment."""
    pass


@ml.group("train", context_settings=cli_context)
def ml_train():
    """Train a model: lgbm, lstm, or transformer."""
    pass


ml_train.add_command(ml_train_lgbm, name="lgbm")
ml_train.add_command(ml_train_lstm, name="lstm")

ml.add_command(ml_dataset)
ml.add_command(ml_wfcv)
ml.add_command(ml_fundamentals)
ml.add_command(ml_backtest)
ml.add_command(ml_score)
ml.add_command(ml_backfill)
ml.add_command(ml_seed_calib, name="seed-calibration-daily")
