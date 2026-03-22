# ML pipeline — dataset construction, training, evaluation, and deployment.
from morningalpha import cli_context
import rich_click as click
from morningalpha.ml.dataset import dataset as ml_dataset
from morningalpha.ml.train import train as ml_train
from morningalpha.fundamentals import fundamentals_cmd as ml_fundamentals
from morningalpha.ml.backtest import backtest as ml_backtest
from morningalpha.ml.score import score as ml_score


@click.group(context_settings=cli_context)
def ml():
    """ML pipeline commands — dataset, training, evaluation, and deployment."""
    pass


ml.add_command(ml_dataset)
ml.add_command(ml_train)
ml.add_command(ml_fundamentals)
ml.add_command(ml_backtest)
ml.add_command(ml_score)
