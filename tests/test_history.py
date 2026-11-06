"""Optimization histories need neither minibatches nor classification metrics."""

from dataclasses import asdict
import json

from experiments.history import ExperimentHistory, HistoryRecord


def test_wpo_objective_and_periodic_evaluation_round_trip(tmp_path):
    history = ExperimentHistory(
        method='wpo', config={'checkpoint': 'mnist.pt'},
        optimization_history=[
            HistoryRecord(metrics={'normalized_weight_error': .25}, step=1, seconds=2.),
            HistoryRecord(metrics={'loss': .1, 'accuracy': .97}, step=1, seconds=2.,
                          count=100, realization_id='validation1', t_inference=86400.),
        ],
    )
    path = tmp_path / 'nested' / 'wpo.json'
    history.save(path)
    saved = json.loads(path.read_text())
    assert saved == asdict(history)
    objective, evaluation = saved['optimization_history']
    assert objective['metrics'] == {'normalized_weight_error': .25}
    assert objective['batch'] is None and objective['epoch'] is None
    assert objective['count'] is None
    assert evaluation['metrics']['accuracy'] == .97
    assert saved['digital_baseline'] is None and saved['digital_history'] == []


def test_separate_experiments_do_not_share_history_lists():
    oi = ExperimentHistory(method='oi', config={})
    wpo = ExperimentHistory(method='wpo', config={})
    oi.optimization_history.append(HistoryRecord(metrics={'loss': .1}))
    oi.results.append({'accuracy': .97})
    oi.digital_history.append({'epoch': 1})
    assert wpo.optimization_history == wpo.results == wpo.digital_history == []
