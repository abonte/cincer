import math
import os.path
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from negsup import load
from scipy import stats

MODEL_LABEL = {
    'logreg': 'LR',
    'kernel_logreg': 'KLR',
    'fullnet': 'FC',
    'convnet': 'CNN',
}

INSPECTOR_LABEL = {
    'never': '⊤',
    'random': 'rand',
    'margin': 'μ',
    'influence': 'IF',
    'fisher': 'fisher'
}

NEGOTIATOR_LABEL = {
    'random': 'rand',
    'nearest': 'NN',
    'if': 'IF',
    'nearest-if': 'NN-IF',
    'practical_fisher': 'Practical Fisher',
    'approx_fisher': 'Diag. Fisher',
    'block_fisher': 'Block Fisher',
    'top_fisher': 'Top Fisher',
    'nearest_fisher': 'NN-Fisher',
    'full_fisher': 'Full Fisher',
    'ce_removal': 'Drop CE'
}

DATASET_LABEL = {
    'mnist': 'mnist',
    'fashion_mnist': 'fashion',
    'breast': 'breast',
    'adult': 'adult',
    '20ng': '20NG'
}

QUESTIONS = {
    'q3': ['nearest', 'practical_fisher', 'top_fisher', 'upper_bound'],
    'q1': ['ce_removal', 'no_ce', 'top_fisher']
}


def _get_style(trace_args, style_by):
    try:
        threshold = trace_args.threshold
    except:
        threshold = None

    # label = MODEL_LABEL[trace_args.model]
    label = ''
    if trace_args.inspector != 'always':
        # label += f' {INSPECTOR_LABEL[trace_args.inspector]}'
        if style_by != 'inspector':
            label += f'{NEGOTIATOR_LABEL[trace_args.negotiator]}'

    if trace_args.inspector == 'always' and trace_args.p_noise == 0.0:
        label = 'No noise'

    if trace_args.no_ce:
        label = 'No CE'
    if trace_args.negotiator == 'nearest_fisher':
        label += f' r={trace_args.nfisher_radius}'

    # label += f' {int(trace_args.p_noise * 100)}%'

    if style_by == 'noise':
        n = trace_args.p_noise
        color = (n, n, n)

    elif style_by == 'threshold':
        t = threshold
        color = (t, t, t)

    elif style_by == 'inspector':
        if trace_args.inspector == 'always':
            color = 'dimgray'
        elif trace_args.inspector == 'random':
            color = 'mediumorchid'
        elif trace_args.inspector == 'margin':
            color = 'lightseagreen'
        elif trace_args.inspector == 'influence':
            color = 'tomato'
        else:
            raise ValueError()

    elif style_by == 'negotiator':
        if trace_args.no_ce:
            color = 'gray'
        elif trace_args.inspector == 'always':
            color = 'darkgray'
        elif trace_args.negotiator == 'random':
            color = 'dimgray'
        elif trace_args.negotiator == 'nearest':
            color = 'limegreen'
        elif trace_args.negotiator == 'practical_fisher':
            color = 'dodgerblue'
        elif trace_args.negotiator == 'approx_fisher':
            color = 'deepskyblue'
        elif trace_args.negotiator == 'block_fisher':
            color = 'blue'
        elif trace_args.negotiator == 'top_fisher':
            color = 'red'  # 'deeppink'
        elif trace_args.negotiator == 'nearest_fisher':
            color = 'violet'
        elif trace_args.negotiator == 'full_fisher':
            color = 'hotpink'
        elif trace_args.negotiator == 'if':
            color = 'darkorchid'
        elif trace_args.negotiator == 'ce_removal':
            color = 'orange'
        else:
            raise ValueError(trace_args.negotiator)

    else:
        raise ValueError()

    marker = {
        'always': '+',
        'never': '+',
        'random': '.',
        'margin': 'o',
        'influence': '*',
        'fisher': 'v'
    }[trace_args.inspector]

    if trace_args.negotiator == 'nearest_fisher' and trace_args.nfisher_radius == 0.10:
        marker = '+'
    elif trace_args.negotiator == 'nearest_fisher' and trace_args.nfisher_radius == 0.25:
        marker = '*'

    if trace_args.p_noise == 0 or trace_args.inspector == 'always':
        linestyle = 'dashed'
    else:
        linestyle = 'solid'

    zorder = {
        'nearest': 1,
        'nearest_fisher': 1,
        'top_fisher': 3,
        'practical_fisher': 2
    }.get(trace_args.negotiator, None)

    return label, color, marker, linestyle, zorder


def to_be_plot(question, trace_args, plot_args):
    if trace_args.negotiator == 'nearest_fisher':
        return False
    method = trace_args.negotiator
    method = 'no_ce' if trace_args.no_ce else method
    method = 'upper_bound' if trace_args.p_noise == 0.0 and trace_args.inspector == 'always' else method
    print(method)
    # plot upper bound only for supplementary
    if method == 'upper_bound' and not plot_args.sup:
        return False
    return method in QUESTIONS[question]


def _draw(plot_args, traces, trace_args, metrics):
    n_pickles, n_repeats, n_iters, n_measures = traces.shape
    fontsize = {
        'xlabel': 20,
        'ylabel': 20,
        'tick': 18,
        'legend': 30
    }

    AX_CONFIGS = {
        'precision': (True, 'Test $Pr$', 'pr', 'lower right'),
        'recall': (True, 'Test $Rc$', 'rc', 'lower right'),
        'n_cleaned': (True, '# cleaned', 'nc', 'upper left'),
        'f1': (True, '$F_1$', 'f1', 'lower right'),
        'n_mistakes_seen': (False, '# Mistakes Seen', 'nm', 'upper left'),
        'n_queried': (True, '# queries', 'nq', 'upper left'),
        'n_cleaned_ce': (True, '# Cleaned ce', '', 'upper left'),
        'ece': (True, 'ECE', 'ece', 'upper left'),
        'zs_value': (True, 'Fisher value', 'fv', 'upper left'),
        'case1': (True, 'user wrong, machine wrong, ce correct', '', 'upper left'),
        'case4': (True, 'user wrong, machine wrong, ce wrong', '', 'upper left'),

        'case2': (True, 'user wrong, machine correct, ce correct', '', 'upper left'),
        'case5': (True, 'user wrong, machine correct, ce wrong', '', 'upper left'),

        'case3': (True, 'user correct, machine wrong, ce correct', '', 'upper left'),
        'case6': (True, 'user correct, machine wrong, ce wrong', '', 'upper left'),

        'case11': (True, 'machine wrong, ce correct', '', 'upper left'),
        'case7': (True, 'machine wrong, ce wrong', '', 'upper left'),

        'case12': (True, 'user wrong, ce correct', '', 'upper left'),
        'case8': (True, 'user wrong, ce wrong', '', 'upper left'),

        'case13': (True, 'machine correct, ce correct', '', 'upper left'),
        'case9': (True, 'machine correct, ce wrong', '', 'upper left'),

        'case14': (True, 'user correct, ce correct', '', 'upper left'),
        'case10': (True, 'user correct, ce wrong', '', 'upper left'),
    }

    if plot_args.summary or plot_args.sup:
        # This is ugly
        if plot_args.sup:
            to_show = ['f1', 'n_cleaned', 'n_queried']
        else:
            to_show = ['f1', 'n_cleaned']
        for metric, config in AX_CONFIGS.items():
            if metric not in to_show:
                mod_config = (False,) + config[1:]
                AX_CONFIGS[metric] = mod_config

    n_plots = len([config for _, config in AX_CONFIGS.items() if config[0]])

    if args.summary or args.sup:
        n_rows, n_cols = n_plots, 1
    else:
        n_rows, n_cols = math.ceil(n_plots / 2), 2

    figsize = (
        math.ceil(n_cols * 6.48),
        math.ceil(n_rows * 4.8 * 0.66),  # Without the 0.66 it'd be 16:9
    )
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    ax_idx = 0
    for metric_name in AX_CONFIGS.keys():
        enabled, name, shorthand, legend_loc = AX_CONFIGS[metric_name]
        if not enabled:
            continue

        ax = axs.flatten()[ax_idx]
        ax_idx += 1

        for p in range(n_pickles):
            if metric_name == 'n_cleaned' and trace_args[p].p_noise == 0.0:
                # do not plot upper bound
                continue
            if not to_be_plot(plot_args.question, trace_args[p], plot_args):
                continue

            if metric_name == 'n_cleaned_ce':
                ax = _plot_line(plot_args, ax, metric_name, metrics, n_iters, p, traces,
                                ' ce')
                ax = _plot_line(plot_args, ax, 'n_cleaned_ex', metrics, n_iters, p,
                                traces, ' ex', '--')
            else:
                ax = _plot_line(plot_args, ax, metric_name, metrics, n_iters, p, traces)

        if metric_name == 'negotiator_value':
            plot_fisher_value(ax, p, traces, metrics, n_iters, )

        if metric_name in ['n_queried', 'n_cleaned']:
            x = np.arange(n_iters)
            y = (x + trace_args[p].p_known) * trace_args[p].p_noise

            # ax.plot(x, y,
            #        linewidth=2,
            #        color='gray',
            #        linestyle='dashed')
            # ax.set_ylabel('#', fontsize=18)

        if (metric_name == 'n_queried' and plot_args.sup) or (
                metric_name == 'n_cleaned' and plot_args.summary):
            ax.set_xlabel('Iterations', fontsize=18)

        if not args.summary and not plot_args.sup:
            ax.set_title(name)
        ax.tick_params(axis='both', labelsize=fontsize['tick'])
        ax.legend(loc=legend_loc, fontsize=15, shadow=False, ncol=2)

    if plot_args.summary or plot_args.sup:
        fig_leg = plt.figure(figsize=(7, 1))
        ax_leg = fig_leg.add_subplot(111)
        # add the legend from the previous axes
        leg = ax_leg.legend(*axs[0].get_legend_handles_labels(), ncol=6,
                            fontsize=fontsize['legend'], facecolor='white')

        # hide the axes frame
        ax_leg.axis('off')
        for line in leg.get_lines():
            line.set_linewidth(7.0)
        fig_leg.savefig(
            os.path.join(plot_args.output_path, f'{plot_args.question}_legend.pdf'),
            bbox_inches='tight')
        for ax in axs.flatten():
            ax.get_legend().remove()

    basename = _get_basename(plot_args, trace_args[1])
    if 'full_fisher' in [a.negotiator for a in trace_args]:
        basename += '__full_fisher'

    fig.savefig(os.path.join(plot_args.output_path, f'{basename}.pdf'),
                bbox_inches='tight',
                pad_inches=0)
    del fig


def _plot_line(plot_args, ax, metric_name, metrics, n_iters, p, traces, end_label='',
               style=None, override_marker=None):
    label, color, marker, linestyle, zorder = _get_style(trace_args[p],
                                                         plot_args.style_by)
    label = 'CINCER (Top Fisher)' if plot_args.question == 'q3' and trace_args[
        p].negotiator == 'top_fisher' else label

    linestyle = linestyle if style is None else style
    marker = marker if override_marker is None else override_marker
    # [pickle, runs, iterations, metrics]
    m = metrics.index(metric_name)
    perf = traces[p, :, :, m]

    perf = perf.astype(np.float)
    x = np.arange(n_iters)
    y = np.mean(perf, axis=0)
    yerr = np.std(perf, axis=0) / np.sqrt(perf.shape[0])

    ax.plot(x, y,
            linewidth=2,
            color=color,
            marker=marker,
            markevery=10,
            linestyle=linestyle,
            label=label + end_label,
            zorder=zorder)
    ax.fill_between(x, y - yerr, y + yerr,
                    color=color,
                    alpha=0.35,
                    linewidth=0,
                    zorder=zorder)

    return ax


def plot_fisher_value(ax, p, traces, metrics, n_iters):
    for run in [0]:  # range(len(perf)):
        m = metrics.index('zs_value')
        perf = traces[p, :, :, m]
        x = np.arange(n_iters)
        y = perf[run, :]

        color = 'C' + str(run)
        ax.plot(x, y,
                linewidth=2,
                color=color,
                markevery=10,
                linestyle='solid',
                label='fisher value')

    m = metrics.index('noisy_ce')
    noisy_ce = traces[p, :, :, m]
    for t in noisy_ce[noisy_ce > 0]:
        ax.axvline(x=t, color=color)

    return ax


def _plot_correlation_influence_and_fisher(plot_args, trace_args, traces):
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    coords_inf = traces['influence']
    coords_fisher = traces['fisher']
    coords_fisher_mst = traces['fisher_mst']

    corr = stats.spearmanr(coords_inf[:, 0], coords_inf[:, 1])
    axs[0].set_title(f'c={corr[0]:.2f} p={corr[1]:.2f}, n={coords_inf.shape[0]}')
    axs[0].scatter(coords_inf[:, 0], coords_inf[:, 1])
    axs[0].set_xlabel(f'true loss diff at test point')
    axs[0].set_ylabel(f'influence est. loss diff at test point')
    # axs[0].set_xlim(-0.1, 0.5)
    # axs[0].set_ylim(-0.5*1e-6, 0.4*1e-6)

    if trace_args.p_noise != 0.0:
        all_coords_fisher = np.vstack([coords_fisher_mst, coords_fisher])
    else:
        all_coords_fisher = coords_fisher

    print(f'{trace_args.dataset} - {trace_args.model} - {trace_args.negotiator}| '
          f'shape Fisher: {all_coords_fisher.shape}, shape IF: {coords_inf.shape}')

    corr = stats.spearmanr(all_coords_fisher[:, 0], all_coords_fisher[:, 1])
    axs[1].set_title(f'c={corr[0]:.2f} p={corr[1]:.2f}, n={all_coords_fisher.shape[0]}')
    axs[1].scatter(coords_fisher[:, 0], coords_fisher[:, 1])
    if trace_args.p_noise != 0.0:
        axs[1].scatter(coords_fisher_mst[:, 0], coords_fisher_mst[:, 1], c='red')
    axs[1].set_xlabel(f'true loss diff at test point')
    axs[1].set_ylabel(f'fisher kernel at test point')
    # axs[1].set_xlim(-0.2, 0.2)
    # axs[1].set_ylim(-1000, 1000)

    basename = _get_basename(plot_args, trace_args) + f'__{trace_args.negotiator}'
    fig.savefig(os.path.join(plot_args.output_path, f'{basename}.pdf'),
                bbox_inches='tight',
                pad_inches=0)


def _plot_ce_precisions(plot_args, traces, trace_args, metrics):
    at_k = [metrics.index('ce_pr_at_5'), metrics.index('ce_pr_at_10')]

    mean, err, count = {}, {}, {}

    for i_neg in range(traces.shape[0]):
        neg = trace_args[i_neg].negotiator
        if neg == 'random' or neg == 'nearest':
            continue
        err[neg], mean[neg], count[neg] = [], [], []

        for i in at_k:
            neg_traces = traces[i_neg, :, :, i]
            mean[neg].append(np.nanmean(neg_traces))
            err[neg].append(np.nanstd(neg_traces) / np.sqrt(
                np.count_nonzero(~np.isnan(neg_traces))))
            count[neg].append(np.count_nonzero(~np.isnan(neg_traces)))


    x = np.arange(len(at_k))
    n_bars = len(err.keys())
    width = 0.25 if n_bars == 3 else 0.20
    length = 3.5
    fig, ax = plt.subplots(figsize=(length, 2.5))
    i = 0
    for neg in ['if', 'practical_fisher', 'top_fisher', 'full_fisher']:
        if neg not in err.keys():
            continue
        i += 1
        tmp_trace_args = copy(trace_args[0])
        tmp_trace_args.negotiator = neg
        _, color, _, _, _ = _get_style(tmp_trace_args, 'negotiator')
        ax.bar(x + i * width, mean[neg], width, yerr=err[neg],
               color=color, label=NEGOTIATOR_LABEL[neg])

    ax.set_ylabel('precision', fontsize=20)
    ax.set_ylim([0, 0.49])
    center = width*2 if n_bars == 3 else (width*2) + width/2
    ax.set_xticks(x + center)
    ax.set_xticklabels(['Pr@5', 'Pr@10'], fontsize=20)
    ax.legend()
    #ax.set_title(
    #    f'{DATASET_LABEL[trace_args[0].dataset]} {MODEL_LABEL[trace_args[0].model]}, n={count["if"][0]}',
    #    fontsize=20)

    fig_leg = plt.figure(figsize=(7, 1))
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    leg = ax_leg.legend(*ax.get_legend_handles_labels(), ncol=1,
                        fontsize=30, facecolor='white')

    # hide the axes frame
    ax_leg.axis('off')
    for line in leg.get_lines():
        line.set_linewidth(7.0)
    fig_leg.savefig(
        os.path.join(plot_args.output_path, f'{n_bars}{plot_args.question}_legend.pdf'),
        bbox_inches='tight')
    ax.get_legend().remove()

    basename = _get_basename(plot_args, trace_args[0]) + '__ce_precision'
    fig.savefig(os.path.join(plot_args.output_path, f'{basename}.pdf'),
                bbox_inches='tight',
                pad_inches=0)


def _get_basename(plot_args, args):
    fields_model = [
        (None, plot_args.question),
        ('t', args.threshold if args.inspector != 'never' else 0.0),
        (None, args.dataset),
        (None, args.model)
    ]
    if plot_args.summary:
        fields_model.append((None, 'summary'))

    basename = '__'.join([name + '=' + str(value) if name else str(value)
                          for name, value in fields_model])

    return basename


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    parser.add_argument('-o', dest='output_path', type=str, default='.',
                        help='output folder')
    parser.add_argument('--question', type=str, default='q3',
                        choices=['q1', 'q3', 'eval_influence', 'eval_ce'])
    parser.add_argument('--summary', action='store_true',
                        help='plot F1 and # cleaned only')
    parser.add_argument('--sup', action='store_true',
                        help='plot F1, # cleaned and # queries only')
    parser.add_argument('--style-by', type=str, choices=[
        'noise', 'threshold', 'inspector', 'negotiator'
    ], help='color plots by threshold rather than by negotiator')
    args = parser.parse_args()

    print(f'question: {args.question}')
    if args.question == 'eval_influence':
        data = load(args.pickles[0])
        _plot_correlation_influence_and_fisher(args, data['args'], data)
    else:
        traces, trace_args, metrics = [], [], None
        for path in args.pickles:
            data = load(path)
            print(
                f'loaded traces of shape {np.array(data["traces"]).shape}, {data["args"].negotiator} | noce={data["args"].no_ce}')
            metrics = data['traces'][0].columns.to_list()
            traces.append(data['traces'])
            trace_args.append(data['args'])

        print(
            f'{trace_args[0].dataset} - {trace_args[1].inspector} - {trace_args[0].model}')

        traces = np.array(traces)

        plt.style.use('ggplot')
        if args.question in ['q1', 'q3']:
            _draw(args, traces, trace_args, metrics)
        elif args.question == 'eval_ce':
            _plot_ce_precisions(args, traces, trace_args, metrics)

        print('==================')
