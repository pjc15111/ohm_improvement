"""Plot saved experiment histories without rerunning training or inference."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


X_LABELS = {'seconds': 'Elapsed optimization time (s)', 'step': 'Optimizer step',
            'epoch': 'Epoch'}
GROUPINGS = {
    'overall': (),
    'by_time': ('t_inference',),
    'by_realization': ('realization_id',),
    'by_realization_and_time': ('realization_id', 't_inference'),
}


def aggregate_samples(samples, group_by=()):
    """Average metrics by optimizer step and selected dimensions, using counts.

    Never aggregate by elapsed time: steps, not timestamps, identify observations
    shared by the realization/time samples in a batch.
    """
    groups = defaultdict(list)
    for sample in samples:
        key = (sample['step'], *(sample[field] for field in group_by))
        groups[key].append(sample)
    records = []
    for group in groups.values():
        count = sum(sample['count'] for sample in group)
        record = {key: group[0][key] for key in ('step', 'seconds', 'epoch', *group_by)}
        record['count'] = count
        for metric in ('loss', 'accuracy'):
            record[metric] = sum(sample[metric] * sample['count'] for sample in group) / count
        records.append(record)
    return sorted(records, key=lambda record: record['step'])


def _label(name, record, group_by):
    parts = [name]
    for field in group_by:
        value = record[field]
        parts.append(f't={value:g}s' if field == 't_inference' else str(value))
    return ', '.join(parts)


def _save_plot(curves, baseline, metric, title, x_axis, path):
    """Draw standalone PNG/PDF figures using a headless canvas."""
    figure = Figure(figsize=(10, 6), layout='constrained')
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for index, (label, records) in enumerate(curves.items()):
        scale = 100 if metric == 'accuracy' else 1
        axis.plot([r[x_axis] for r in records], [r[metric] * scale for r in records],
                  marker='.', markersize=4, label=label, color=colormaps['tab20'](index % 20),
                  linestyle='-' if index < 20 else '--')
    baseline_value = baseline[metric] * (100 if metric == 'accuracy' else 1)
    axis.axhline(baseline_value, linestyle=':', color='black', linewidth=1.5,
                 label=f'Digital {metric} baseline ({baseline_value:.4g})')
    axis.set(title=title, xlabel=X_LABELS.get(x_axis, 'Elapsed time since programming (s)'),
             ylabel='Accuracy (%)' if metric == 'accuracy' else 'Mean classification loss')
    if metric == 'accuracy':
        axis.set_ylim(0, 100)
    if x_axis == 't_inference':
        axis.set_xscale('symlog', linthresh=1)
    handles, labels = axis.get_legend_handles_labels()
    if metric == 'loss':
        accuracy_axis = axis.twinx()
        accuracy_axis.set_ylabel('Digital reference accuracy (%)', color='dimgray')
        accuracy_axis.set_ylim(0, 100)
        accuracy_axis.tick_params(axis='y', colors='dimgray')
        reference = accuracy_axis.axhline(
            baseline['accuracy'] * 100, linestyle=':', color='dimgray',
            linewidth=1.5, label=f"Digital accuracy ({baseline['accuracy'] * 100:.2f}%)",
        )
        handles.append(reference)
        labels.append(reference.get_label())
    axis.grid(alpha=.25)
    if len(handles) > 8:
        axis.legend(handles, labels, fontsize='small', loc='upper center',
                    bbox_to_anchor=(.5, -.16), ncol=3)
    else:
        axis.legend(handles, labels, fontsize='small', loc='best')
    paths = []
    for suffix in ('.png', '.pdf'):
        output = path.with_suffix(suffix)
        figure.savefig(output, dpi=160)
        paths.append(output)
    figure.clear()
    return paths


def plot_histories(history, output_dir, *, x_axis='seconds'):
    """Generate two digital and eight optimization plots, plus retention curves.

    Optimization curves describe training minibatches before each update; they
    are not held-out evaluations. Final retention curves use stock evaluation.
    Additional methods (including future WPO results) are plotted when present.
    """
    if x_axis not in X_LABELS:
        raise ValueError(f'Unknown history x-axis: {x_axis}')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    baseline = history['digital_baseline']
    digital = history['digital_history']
    training = [batch for epoch in digital for batch in epoch['training']['batches']]
    evaluation = [dict(epoch['evaluation'], epoch=epoch['epoch'], seconds=epoch['seconds'],
                       step=epoch['step']) for epoch in digital]
    histories = history['optimization_histories'] if 'optimization_histories' in history else {'OI': history['oi_history']}
    paths = []
    for metric in ('loss', 'accuracy'):
        paths.extend(_save_plot({'Training': training, 'Evaluation': evaluation}, baseline,
                                metric, f'Digital training {metric}', x_axis,
                                directory / f'digital_{metric}'))
        for suffix, group_by in GROUPINGS.items():
            curves = defaultdict(list)
            for method, epochs in histories.items():
                samples = [sample for epoch in epochs for sample in epoch['samples']]
                for record in aggregate_samples(samples, group_by):
                    curves[_label(method, record, group_by)].append(record)
            title = f'Optimization training {metric}: {suffix.replace("_", " ")}'
            paths.extend(_save_plot(curves, baseline, metric, title, x_axis,
                                    directory / f'optimization_{metric}_{suffix}'))
        # Include every configured method automatically; no WPO implementation
        # or fabricated WPO curve is needed here.
        retention = defaultdict(list)
        grouped = defaultdict(list)
        for record in history['results']:
            grouped[(record['method'], record['t_inference'])].append(record)
        for (method, time), records in sorted(grouped.items()):
            count = sum(record['count'] for record in records)
            retention[method].append(dict(
                t_inference=time,
                **{m: sum(r[m] * r['count'] for r in records) / count
                   for m in ('loss', 'accuracy')},
            ))
        paths.extend(_save_plot(retention, baseline, metric, f'Stock inference {metric}',
                                't_inference', directory / f'inference_{metric}'))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--x-axis', choices=list(X_LABELS), default='seconds')
    args = parser.parse_args()
    history = json.loads(args.history.read_text())
    plot_histories(history, args.output_dir or args.history.parent, x_axis=args.x_axis)


if __name__ == '__main__':
    main()
