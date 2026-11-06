import pytest

from experiments import plotting


def make_samples():
    return [dict(step=step, seconds=step * 2., epoch=step / 2.,
                 realization_id=realization, t_inference=time, count=count,
                 loss=loss, accuracy=1 - loss / 10)
            for step in (1, 2)
            for realization, count in [('train1', 1), ('train2', 3)]
            for time, loss in [(0., 2.), (86400., 4.)]]


def test_history_groupings_and_weighted_means():
    samples = make_samples()
    for name, fields in plotting.GROUPINGS.items():
        records = plotting.aggregate_samples(samples, fields)
        assert len(records) == {'overall': 2, 'by_time': 4, 'by_realization': 4,
                                'by_realization_and_time': 8}[name]
    # Sample counts weight the mean, even for manually combined histories.
    samples[0]['loss'] = 10.
    records = plotting.aggregate_samples(samples)
    assert records[0]['loss'] == pytest.approx((10 + 4 + 3 * 2 + 3 * 4) / 8)
    assert records[0]['count'] == 8
    assert records[1]['step'] == 2


def test_plot_dispatch_includes_all_methods_and_baselines(monkeypatch, tmp_path):
    calls = []

    def capture(curves, baseline, metric, title, x_axis, path):
        calls.append((curves, baseline, metric, path.name))
        return []

    monkeypatch.setattr(plotting, '_save_plot', capture)
    samples = make_samples()
    baseline = dict(loss=.5, accuracy=.95)
    history = dict(digital_baseline=baseline,
                   digital_history=[dict(epoch=1, step=2, seconds=4.,
                                         training=dict(batches=samples), evaluation=baseline)],
                   optimization_histories={'OI': [dict(samples=samples)],
                                           'Another configuration': [dict(samples=samples)]},
                   results=[dict(method=method, t_inference=time, count=10, **baseline)
                            for method in ['baseline', 'oi', 'wpo'] for time in [0., 86400.]])
    plotting.plot_histories(history, tmp_path, x_axis='step')
    assert len(calls) == 12
    assert all(b == baseline for _, b, _, _ in calls)
    for curves, _, metric, name in calls:
        if name.startswith('inference'):
            assert set(curves) == {'baseline', 'oi', 'wpo'}
        elif name.endswith('overall'):
            assert set(curves) == {'OI', 'Another configuration'}
