"""Step 34: descriptive saved-artifact analysis only. No data loader or estimator fit."""
from pathlib import Path
import sys
import os
import json
import hashlib
import math
import platform
import warnings
from datetime import datetime, timezone
from itertools import combinations

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = 'imbalance_cv_step33_20261109T165943Z'
SOURCE = ROOT / 'results/imbalance_cv' / RUN
# Keep the experiment environment unchanged. Plot-only dependencies are local.
os.environ['MPLCONFIGDIR'] = str(OUT / '_matplotlib_config')
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                            matthews_corrcoef, brier_score_loss)
sys.path.append(str(OUT / '_plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

METRICS = ['roc_auc', 'average_precision', 'recall_sensitivity', 'specificity',
           'precision', 'f1', 'mcc', 'balanced_accuracy', 'brier_score']
LABELS = dict(zip(METRICS, ['ROC-AUC', 'AP', 'Sensitivity', 'Specificity', 'Precision',
                          'F1', 'MCC', 'Balanced accuracy', 'Brier score']))
MODELS = ['dummy_prior', 'logistic_regression', 'decision_tree', 'random_forest', 'svc_rbf']
MODEL_NAMES = dict(zip(MODELS, ['Dummy', 'Logistic regression', 'Decision tree', 'Random forest', 'RBF SVM']))
STRATEGIES = ['none', 'class_weight_balanced', 'random_over_sampler', 'random_under_sampler', 'smote', 'adasyn']
SHORT = dict(zip(STRATEGIES, ['None', 'Weight', 'ROS', 'RUS', 'SMOTENC', 'CatADASYN']))
COLORS = dict(zip(STRATEGIES, ['#676767', '#0072B2', '#009E73', '#E69F00', '#CC79A7', '#D55E00']))
MARKERS = dict(zip(STRATEGIES, ['o', 's', '^', 'v', 'D', 'P']))
KEYS = ['pipeline', 'model', 'strategy']
ORDER = [f'{m}__{s}' for m in MODELS for s in STRATEGIES if m != 'dummy_prior' or s == 'none']
CHECKS = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(name, passed, **details):
    CHECKS.append(dict(check=name, passed=bool(passed), **details))
    if not passed:
        raise AssertionError(f'{name}: {details}')


def read(stem):
    return pd.read_csv(SOURCE / f'{RUN}_{stem}.csv', float_precision='round_trip')


def write_csv(frame, stem):
    frame.to_csv(OUT / f'{stem}.csv', index=False, float_format='%.15g')


def summarize(frame, scope):
    records = []
    for key, group in frame.groupby(KEYS, sort=False):
        for metric in METRICS:
            v = group[metric]
            records.append(dict(zip(KEYS, key)) | dict(scope=scope, metric=metric,
                mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()),
                max=float(v.max()), n_units=len(v)))
    return pd.DataFrame(records)


def wide_summary(long):
    rows = []
    for key in ORDER:
        group = long[long.pipeline == key]
        record = {k: group.iloc[0][k] for k in KEYS}
        record['scope'] = group.iloc[0]['scope']
        record['n_units'] = int(group.iloc[0]['n_units'])
        for row in group.itertuples():
            for statistic in ['mean', 'sd', 'min', 'max']:
                record[f'{row.metric}_{statistic}'] = getattr(row, statistic)
        rows.append(record)
    return pd.DataFrame(rows)


def mcc_counts(tn, fp, fn, tp):
    denominator = math.sqrt(float(tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    return (float(tp*tn-fp*fn)/denominator if denominator else 0.0), bool(denominator)


def score(group):
    y = group.y_true.to_numpy(dtype=int)
    p = group.y_proba.to_numpy(dtype=float)
    predicted = (p >= 0.5).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp) if tp+fp else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    mcc, defined = mcc_counts(tn, fp, fn, tp)
    check('MCC formula matches sklearn', abs(mcc - matthews_corrcoef(y, predicted)) < 1e-12)
    return dict(roc_auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
                recall_sensitivity=recall, specificity=specificity, precision=precision,
                f1=f1, mcc=mcc, balanced_accuracy=(recall+specificity)/2,
                brier_score=brier_score_loss(y, p), tn=tn, fp=fp, fn=fn, tp=tp,
                precision_defined=bool(tp+fp), mcc_denominator_nonzero=defined,
                predicted_positive=tp+fp, n_samples=len(y), n_positive=int(y.sum()))


def md_table(frame, columns, names=None, decimals=4):
    names = names or columns
    rows = ['| ' + ' | '.join(names) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f'{value:.{decimals}f}' if isinstance(value, (float, np.floating)) else str(value))
        rows.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(rows)


def pretty(key):
    model, strategy = key.split('__')
    return f'{MODEL_NAMES[model]} / {SHORT[strategy]}'


def savefig(fig, name, caption):
    fig.savefig(OUT / f'{name}.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    FIGURES.append(dict(file=f'{name}.png', caption=caption))


def main():
    inputs = {p.name: sha(p) for p in SOURCE.iterdir() if p.is_file()}
    manifest = json.loads((SOURCE / f'{RUN}_manifest.json').read_text())
    verification = json.loads((SOURCE / f'{RUN}_verification.json').read_text())
    independent = json.loads((SOURCE / f'{RUN}_independent_verification.json').read_text())
    check('Step33 core verdict', verification['overall_passed'] and verification['n_failed'] == 0)
    check('Step33 independent verdict', independent['overall_passed'] and independent['n_failed'] == 0)
    for key, entry in manifest['files'].items():
        path = ROOT / entry['path']
        check(f'core artifact {key} present', path.is_file())
        if entry['sha256']:
            check(f'core artifact {key} SHA256', sha(path) == entry['sha256'])
    for key, entry in independent['supplemental_integrity'].items():
        check(f'independent input {key} SHA256', sha(entry['path']) == entry['sha256'])
    fold = read('fold_metrics')
    oof = read('oof_predictions')
    saved_summary = read('summary')
    saved_wide = read('summary_wide')
    saved_pooled = read('pooled_by_repeat')
    sampling = read('actual_sampling_ratios')
    check('625 folds / 25 configurations', len(fold) == 625 and set(fold.pipeline) == set(ORDER))
    check('511000 OOF / five repeats', len(oof) == 511000 and set(oof.repeat) == set(range(1, 6)))
    check('no duplicate OOF', not oof.duplicated(['pipeline', 'repeat', 'row_index']).any())
    check('finite bounded OOF', np.isfinite(oof.y_proba).all() and oof.y_proba.between(0, 1).all())
    check('fixed saved threshold', fold.threshold.eq(0.5).all() and saved_pooled.threshold.eq(0.5).all())
    coverage = []
    reference = oof[(oof.pipeline == ORDER[0]) & (oof.repeat == 1)].set_index('row_index').y_true.sort_index()
    check('4088 training identities / 199 positives', len(reference) == 4088 and int(reference.sum()) == 199)
    pooled_records = []
    derived_fold_records = []
    largest_saved_deviation = 0.0
    for key in ORDER:
        group = oof[oof.pipeline == key]
        metrics = fold[fold.pipeline == key]
        check(f'{key} 25 fold identities', len(metrics) == 25 and set(zip(metrics.repeat, metrics.fold)) ==
              {(r, f) for r in range(1, 6) for f in range(1, 6)})
        model, strategy = key.split('__')
        for repeat in range(1, 6):
            rows = group[group.repeat == repeat]
            labels = rows.set_index('row_index').y_true.sort_index()
            ok = len(rows) == 4088 and labels.equals(reference) and rows.row_index.is_unique
            check(f'{key} repeat {repeat} coverage', ok)
            coverage.append(dict(pipeline=key, repeat=repeat, n_oof=len(rows), positives=int(rows.y_true.sum()), passed=ok))
            values = score(rows)
            pooled_records.append(dict(pipeline=key, model=model, strategy=strategy, repeat=repeat,
                                       scope='pooled_within_repeat', **values))
            saved = saved_pooled[(saved_pooled.pipeline == key) & (saved_pooled.repeat == repeat)].iloc[0]
            for metric in METRICS:
                if metric != 'mcc':
                    largest_saved_deviation = max(largest_saved_deviation, abs(values[metric]-saved[metric]))
            for f in range(1, 6):
                part = rows[rows.fold == f]
                values_fold = score(part)
                recorded = metrics[(metrics.repeat == repeat) & (metrics.fold == f)].iloc[0]
                for metric in METRICS:
                    if metric != 'mcc':
                        largest_saved_deviation = max(largest_saved_deviation, abs(values_fold[metric]-recorded[metric]))
                check(f'{key} r{repeat}f{f} confusion counts', all(int(recorded[k]) == values_fold[k] for k in ['tn','fp','fn','tp']))
                derived_fold_records.append(dict(pipeline=key, model=model, strategy=strategy, repeat=repeat,
                                                 fold=f, scope='fold_level', **values_fold))
    check('saved fold / pooled scores reproduced', largest_saved_deviation < 1e-12,
          maximum_absolute_deviation=largest_saved_deviation)
    derived = pd.DataFrame(derived_fold_records)
    pooled = pd.DataFrame(pooled_records)
    fold_long = summarize(derived, 'fold_level_25_correlated_folds')
    repeat_long = summarize(pooled, 'pooled_within_repeat_5_correlated_repeats')
    fw, rw = wide_summary(fold_long), wide_summary(repeat_long)
    for record in saved_summary.itertuples():
        row = fold_long[(fold_long.pipeline == record.pipeline) & (fold_long.metric == record.metric)].iloc[0]
        check('saved long summary reproduced', np.isclose(row['mean'], record.mean, atol=1e-12, rtol=0)
              and np.isclose(row['sd'], record.sd, atol=1e-12, rtol=0))
    for record in saved_wide.to_dict('records'):
        row = fw[fw.pipeline == record['pipeline']].iloc[0]
        check('saved wide summary reproduced', all(abs(row[m+'_mean']-record[m]) < 1e-12 and
              abs(row[m+'_sd']-record[m+'_sd']) < 1e-12 for m in METRICS if m != 'mcc'))
    write_csv(derived, 'fold_metrics_with_mcc')
    write_csv(fw, 'comparison_fold_mean_sd')
    write_csv(rw, 'comparison_pooled_repeat_mean_sd')
    write_csv(pd.concat([fold_long, repeat_long], ignore_index=True), 'comparison_all_metrics_long')
    write_csv(pooled, 'pooled_metrics_by_repeat_with_mcc')
    write_csv(pd.DataFrame(coverage), 'coverage_verification')
    within = derived.groupby(KEYS+['repeat'])[METRICS].agg(['mean','std']).reset_index()
    within.columns = ['_'.join(c).rstrip('_') if isinstance(c, tuple) else c for c in within.columns]
    write_csv(within, 'fold_mean_and_sd_within_each_repeat')
    variability = fold_long.merge(repeat_long, on=KEYS+['metric'], suffixes=('_fold', '_pooled_repeat'))
    write_csv(variability, 'fold_vs_pooled_repeat_variability')
    effects = []
    fold_effects = []
    for frame, output, pairing in [(pooled, effects, ['repeat']), (derived, fold_effects, ['repeat','fold'])]:
        for model in MODELS[1:]:
            base = frame[(frame.model == model) & (frame.strategy == 'none')].set_index(pairing)
            for strategy in STRATEGIES[1:]:
                part = frame[(frame.model == model) & (frame.strategy == strategy)].set_index(pairing)
                for index, record in part.iterrows():
                    pair = index if isinstance(index, tuple) else (index,)
                    row = dict(pipeline=f'{model}__{strategy}', model=model, strategy=strategy,
                               reference_pipeline=f'{model}__none', **dict(zip(pairing, pair)))
                    for metric in METRICS:
                        row[f'delta_{metric}'] = float(record[metric]-base.loc[index, metric])
                    output.append(row)
    effects, fold_effects = pd.DataFrame(effects), pd.DataFrame(fold_effects)
    effect_summary = effects.groupby(KEYS, sort=False)[['delta_'+m for m in METRICS]].agg(['mean','std','min','max']).reset_index()
    effect_summary.columns = ['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in effect_summary.columns]
    write_csv(effects, 'paired_strategy_effects_by_repeat')
    write_csv(fold_effects, 'paired_strategy_effects_by_fold')
    write_csv(effect_summary, 'strategy_effects_vs_none_summary')
    ranks = []
    for strategy in STRATEGIES:
        part = rw[(rw.model != 'dummy_prior') & (rw.strategy == strategy)]
        for metric in METRICS:
            ranking = part[f'{metric}_mean'].rank(ascending=metric == 'brier_score', method='min')
            for idx, row in part.iterrows():
                ranks.append(dict(pipeline=row.pipeline, model=row.model, strategy=strategy,
                                  metric=metric, value=row[f'{metric}_mean'], descriptive_rank=ranking.loc[idx]))
    ranks = pd.DataFrame(ranks)
    write_csv(ranks, 'descriptive_model_ranks_within_strategy')
    reversals = []
    for metric in METRICS:
        for first, second in combinations(MODELS[1:], 2):
            diffs = []
            for strategy in STRATEGIES:
                a = rw[(rw.model == first) & (rw.strategy == strategy)].iloc[0][metric+'_mean']
                b = rw[(rw.model == second) & (rw.strategy == strategy)].iloc[0][metric+'_mean']
                diffs.append((strategy, float(a-b)))
            positive = [(s,d) for s,d in diffs if d > 1e-12]
            negative = [(s,d) for s,d in diffs if d < -1e-12]
            if positive and negative:
                reversals.append(dict(metric=metric, model_a=first, model_b=second,
                    strategy_a_above=positive[0][0], difference_a_above=positive[0][1],
                    strategy_a_below=negative[0][0], difference_a_below=negative[0][1]))
    reversals = pd.DataFrame(reversals)
    write_csv(reversals, 'descriptive_rank_reversals')
    eligibility = []
    for key in ORDER:
        part = derived[derived.pipeline == key]
        eligibility.append(dict(pipeline=key, n_folds=25,
            folds_no_predicted_positives=int((~part.precision_defined).sum()),
            folds_mcc_denominator_zero=int((~part.mcc_denominator_nonzero).sum())))
    write_csv(pd.DataFrame(eligibility), 'metric_definition_diagnostics')
    candidates = {}
    for model in MODELS[1:]:
        part = rw[rw.model == model]
        for metric, rationale in [('average_precision','largest descriptive AP within this family'),
                                  ('mcc','largest descriptive MCC within this family')]:
            chosen = part.loc[part[metric+'_mean'].idxmax()]
            candidates.setdefault(chosen.pipeline, []).append(rationale)
        candidates.setdefault(f'{model}__none', []).append('unchanged-prevalence reference and probability-score comparator')
    shortlist = rw[rw.pipeline.isin(candidates)].copy()
    shortlist['reason_for_later_examination'] = shortlist.pipeline.map(lambda k: '; '.join(candidates[k]))
    write_csv(shortlist, 'configurations_for_subsequent_examination')
    # Fixed order, accessible strategy colors, static 300 dpi scientific plots.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':300})
    means = rw.set_index('pipeline').loc[ORDER]
    names = [pretty(k) for k in ORDER]
    yloc = np.arange(25)
    fig, axes = plt.subplots(1, 9, figsize=(15, 12), sharey=True, gridspec_kw={'wspace':0.15})
    for ax, metric in zip(axes, METRICS):
        vals = means[metric+'_mean'].to_numpy()
        ax.imshow(vals[:,None], aspect='auto', cmap='Blues_r' if metric == 'brier_score' else 'Blues',
                  norm=Normalize(-0.01 if metric == 'mcc' else 0,
                                 0.30 if metric == 'mcc' else (0.25 if metric in ('average_precision','brier_score') else 1)))
        for i, val in enumerate(vals):
            color = 'white' if ((val > 0.16 if metric in ('average_precision','mcc') else val > 0.65) and metric != 'brier_score') else '#111111'
            ax.text(0, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)
        ax.set_xticks([0], [LABELS[metric]], rotation=65, ha='right')
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(yloc, names)
    fig.suptitle('StrokeSure: all configurations and nine performance measures', fontsize=15, y=0.98)
    fig.text(0.03,0.015,'Pooled within-repeat means over 5 repeats. Darker = larger score except Brier (lower is better).\nAP and Brier scales: 0–0.25; MCC: −0.01–0.30; others: 0–1. Compare values across metrics, not colors.',fontsize=9)
    fig.subplots_adjust(left=0.31, bottom=0.11, top=0.94)
    savefig(fig,'figure01_overall_comparison','All 25 configurations; pooled-within-repeat mean of each metric. Brier is a probability score, not a calibration-only measure.')
    fig, axes = plt.subplots(1,2,figsize=(13,11),sharey=True)
    for ax, metric in zip(axes,['roc_auc','average_precision']):
        for i,key in enumerate(ORDER):
            row=means.loc[key]
            ax.errorbar(row[metric+'_mean'],i,xerr=row[metric+'_sd'],fmt='o',markersize=4,
                        capsize=2,color=COLORS[row.strategy])
        ax.set_xlabel(LABELS[metric]); ax.grid(axis='x',alpha=.2)
        ax.set_title(LABELS[metric]+' across configurations')
    axes[0].set_yticks(yloc,names); axes[0].invert_yaxis()
    axes[1].axvline(199/4088,color='#999999',linestyle=':',label='Training prevalence')
    axes[1].legend(loc='lower right')
    fig.suptitle('Discrimination: pooled-within-repeat means ± descriptive SD',fontsize=14)
    fig.text(.3,.015,'SD over five dependent repeats; error bars are not confidence intervals.',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96))
    savefig(fig,'figure02_roc_auc_and_ap','Mean ± sample SD of five pooled repeat scores; AP is average_precision_score, not trapezoidal PR-AUC.')
    legend=[Line2D([0],[0],marker='o',color=COLORS[s],label=SHORT[s],linestyle='') for s in STRATEGIES]
    scatter_legend=[Line2D([0],[0],marker=MARKERS[s],color=COLORS[s],label=SHORT[s],linestyle='') for s in STRATEGIES]
    for metric, fname, title in [('precision','figure03_recall_precision','Recall–precision trade-offs'),
                               ('specificity','figure04_sensitivity_specificity','Sensitivity–specificity trade-offs')]:
        fig, axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True)
        for ax, model in zip(axes.flat,MODELS[1:]):
            for j,strategy in enumerate(STRATEGIES):
                key=f'{model}__{strategy}'; row=means.loc[key]; part=pooled[pooled.pipeline==key]
                ax.scatter(part.recall_sensitivity,part[metric],color=COLORS[strategy],alpha=.25,s=15)
                ax.scatter(row.recall_sensitivity_mean,row[metric+'_mean'],color=COLORS[strategy],
                           marker=MARKERS[strategy],s=75 if strategy=='none' else 45,
                           edgecolor='white',zorder=3)
            if model=='svc_rbf':
                anchor=means.loc['svc_rbf__class_weight_balanced']
                ax.annotate('None / Weight overlap', (anchor.recall_sensitivity_mean,anchor[metric+'_mean']),
                            xytext=(8,9 if metric=='precision' else -17),textcoords='offset points',fontsize=8)
            ax.set_title(MODEL_NAMES[model]); ax.grid(alpha=.2)
            ax.set_xlabel('Sensitivity / recall'); ax.set_ylabel(LABELS[metric]); ax.set_xlim(-.025,.9)
            ax.set_ylim((-.015,max(.25,float(pooled.precision.max())+.03)) if metric=='precision' else (.55,1.025))
        fig.suptitle(title+' at the fixed 0.5 threshold',fontsize=14)
        fig.legend(handles=scatter_legend,loc='lower center',ncol=6,frameon=False)
        fig.tight_layout(rect=(0,.05,1,.95))
        savefig(fig,fname,'Small translucent points are individual repeats; solid points are means. Undefined precision for no positive predictions is conventionally reported as zero and flagged in metric_definition_diagnostics.csv.')
    fig,axes=plt.subplots(4,3,figsize=(13,12),sharex=True,sharey='col')
    for i,model in enumerate(MODELS[1:]):
        for j,metric in enumerate(['average_precision','recall_sensitivity','brier_score']):
            ax=axes[i,j]
            for strategy in STRATEGIES:
                part=pooled[(pooled.model==model)&(pooled.strategy==strategy)].sort_values('repeat')
                ax.plot(part.repeat,part[metric],marker='o',markersize=3,linewidth=1,color=COLORS[strategy])
            ax.grid(alpha=.2); ax.set_xticks(range(1,6)); ax.set_title(MODEL_NAMES[model]+' — '+LABELS[metric],fontsize=10)
            if i==3: ax.set_xlabel('CV repeat (shared random-state sequence 42)')
            if j==0: ax.set_ylabel('Score')
    fig.suptitle('Descriptive variation of pooled OOF scores across repeats',fontsize=14)
    fig.legend(handles=legend,loc='lower center',ncol=6,frameon=False)
    fig.tight_layout(rect=(0,.05,1,.96))
    savefig(fig,'figure05_repeat_variability','Each point pools all five held-out CV folds of one repeat; lines connect split realizations, not time or independent random-seed experiments.')
    fig,axes=plt.subplots(4,3,figsize=(13,12),sharey=True,sharex='col')
    for i,model in enumerate(MODELS[1:]):
        for j,metric in enumerate(['average_precision','recall_sensitivity','specificity']):
            ax=axes[i,j]
            for k,strategy in enumerate(STRATEGIES[1:]):
                row=effect_summary[(effect_summary.model==model)&(effect_summary.strategy==strategy)].iloc[0]
                ax.errorbar(row['delta_'+metric+'_mean'],k,xerr=row['delta_'+metric+'_std'],fmt='o',capsize=3,color=COLORS[strategy])
            ax.axvline(0,color='#777777',linestyle='--',linewidth=.8); ax.grid(axis='x',alpha=.2)
            ax.set_title(MODEL_NAMES[model]+' — Δ '+LABELS[metric],fontsize=10)
            ax.set_yticks(range(5),[SHORT[s] for s in STRATEGIES[1:]])
            if i==3: ax.set_xlabel('Strategy minus none (absolute score units)')
    fig.suptitle('Within-model strategy effects relative to no correction',fontsize=14)
    fig.text(.08,.008,'Paired by repeat; means ± SD of five repeat differences. Descriptive only, not confidence intervals or significance tests.',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96))
    savefig(fig,'figure06_strategy_effects','Positive Δ sensitivity can accompany negative Δ specificity or AP. All nine metric deltas are provided in the CSV tables.')
    # Evidence-grounded report built directly from the computed tables.
    sections = [
        '# StrokeSure Step 34: descriptive analysis of full imbalance CV',
        f'Analysis date: 2026-11-10 (Asia/Calcutta). Source run: `{RUN}`.',
        '## Scope, provenance and verification',
        'Only saved Step 33 artifacts were read. No raw dataset or held-out partition was loaded, and no estimator was created or fitted. Original source artifacts were hashed before and after analysis. The experiment used 4,088 training observations (199 positives; prevalence 0.048679), 25 supported configurations and 5-fold × 5-repeat stratified CV, random state 42, requested sampling ratio 0.5, five neighbors and a fixed 0.5 evaluation threshold. Dummy supports only none; the other four models support all six strategies.',
        'The recorded Step 33 verdicts are 16/16 core gates and 1,457/1,457 independent checks. This analysis independently verifies all configuration/repeat/fold coverage, hashes, scores and saved summaries. There are 625 fold records and 511,000 OOF predictions. Every pipeline contributes 4,088 predictions per repeat; all five repeats together contain repeated predictions of the same patients, not 20,440 independent patients.',
        'SMOTE labels denote SMOTENC. CatADASYN denotes the custom fold-standardized continuous / unit one-hot categorical-aware ADASYN, inverse-transforming continuous values and decoding categories by argmax. It is not stock ADASYN. Its recorded achieved sampling ratios range from 0.506911 to 0.522662 because of per-seed rounding. These realized ratios are a comparison caveat, not failures.',
        '## Metric and aggregation definitions',
        'Primary descriptive comparisons below use the arithmetic mean and sample SD (ddof=1) of five **pooled-within-repeat** metrics. Each repeat pools its five validation folds (4,088 observations) before computing scores. Separately, fold summaries average 25 smaller validation-fold metrics (817 or 818 observations). Averaging fold AP/AUC/F1/MCC is not equivalent to computing them on pooled predictions; even pooled ranking metrics compare scores from distinct fitted folds. Neither summary is an independent-sample confidence interval.',
        'AP is sklearn average_precision_score, not trapezoidal PR-AUC. Sensitivity, specificity, precision, F1, MCC and balanced accuracy use saved probabilities ≥0.5. MCC = (TP·TN−FP·FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)); when its denominator is zero, sklearn convention returns 0 and the condition is flagged. Precision with no predicted positives is similarly flagged and reported as 0 by convention; it should not be interpreted as an observed zero positive predictive value among positive predictions.',
        'Brier score is mean squared probability error (lower is better). It combines reliability/calibration and discrimination-related components, so Brier differences alone do not establish calibration quality. Dedicated calibration assessment remains pending Step 35. No predictions were averaged across repeats to create an ensemble, and no repeats were pooled into one oversized patient sample.',
        '## Main descriptive findings',
    ]
    best_ap=rw.loc[rw.average_precision_mean.idxmax()]
    best_mcc=rw.loc[rw.mcc_mean.idxmax()]
    lowest_brier=rw.loc[rw.brier_score_mean.idxmin()]
    weight=means.loc['logistic_regression__class_weight_balanced']
    none=means.loc['logistic_regression__none']
    confusion=pooled[pooled.pipeline=='logistic_regression__class_weight_balanced'][['tp','fp','fn','tn']].mean()
    fold_ap=fw.set_index('pipeline').loc['logistic_regression__class_weight_balanced']
    sections += [
        f'- The largest descriptive mean AP is {pretty(best_ap.pipeline)}: {best_ap.average_precision_mean:.4f} ± {best_ap.average_precision_sd:.4f}. This is an observed maximum among 25 configurations, not an established winner or unbiased post-selection performance estimate.',
        f'- The largest descriptive MCC is {pretty(best_mcc.pipeline)}: {best_mcc.mcc_mean:.4f} ± {best_mcc.mcc_sd:.4f}. For class-weighted Logistic Regression, sensitivity is {weight.recall_sensitivity_mean:.4f}, specificity {weight.specificity_mean:.4f}, precision {weight.precision_mean:.4f}, and Brier {weight.brier_score_mean:.4f}. Relative to none, sensitivity changes by {weight.recall_sensitivity_mean-none.recall_sensitivity_mean:+.4f}, specificity by {weight.specificity_mean-none.specificity_mean:+.4f}, AP by {weight.average_precision_mean-none.average_precision_mean:+.4f}, and Brier by {weight.brier_score_mean-none.brier_score_mean:+.4f}.',
        f'- The same weighted configuration averages {confusion.tp:.1f} true positives and {confusion.fp:.1f} false positives per 4,088-row repeat, with {confusion.fn:.1f} false negatives. These are means of repeated predictions on the same cohort, not independent patient counts. Sensitivity gains have a substantial false-positive burden.',
        f'- Lowest descriptive Brier is {pretty(lowest_brier.pipeline)}: {lowest_brier.brier_score_mean:.4f} ± {lowest_brier.brier_score_sd:.4f}, but its threshold-0.5 recall is {lowest_brier.recall_sensitivity_mean:.4f}. Thus lower probability error is not equivalent to better minority detection at the fixed threshold.',
        '- Uncorrected Logistic Regression and Random Forest have zero recall throughout these pooled repeats. Class-weighted RBF SVM also has zero recall despite a mean ROC-AUC of 0.8015. Ranking performance does not establish usefulness at a particular probability threshold.',
        '- SMOTENC and CatADASYN lower mean pooled AP relative to none in Logistic Regression, Decision Tree and Random Forest; both increase it in RBF SVM. Synthetic sampling is not uniformly beneficial in these observed configurations. These comparisons do not isolate sampler geometry from realized class ratio or model interactions.',
        f'- For weighted Logistic Regression, fold AP mean ± SD is {fold_ap.average_precision_mean:.4f} ± {fold_ap.average_precision_sd:.4f}; pooled-repeat AP is {weight.average_precision_mean:.4f} ± {weight.average_precision_sd:.4f}. Their different units and aggregation explain why they must not be presented interchangeably.',
        '### Dummy and pooled-ranking interpretation',
        f'Dummy has fold ROC-AUC exactly 0.5 and mean fold AP {fw.iloc[0].average_precision_mean:.6f}. Its pooled-repeat ROC-AUC is {means.loc["dummy_prior__none","roc_auc_mean"]:.6f} and AP {means.loc["dummy_prior__none","average_precision_mean"]:.6f}. The fitted prior is constant within a fold but differs slightly across training folds; pooling produces cross-fold score ordering. This is not evidence of substantive below-chance discrimination and is not a reason to overwrite the recorded scores. Training prevalence {199/4088:.6f} is the theoretical constant-score AP reference. Pooled AUC/AP for all configurations likewise compare scores from distinct fold-fitted estimators; both pooled-repeat and fold summaries are therefore retained.',
        '## Overall comparison: five pooled repeats',
    ]
    for selected,title in [(['roc_auc','average_precision','mcc','brier_score'],'Discrimination, MCC and probability error'),
                           (['recall_sensitivity','specificity','precision','f1','balanced_accuracy'],'Detection trade-offs at threshold 0.5')]:
        table=pd.DataFrame({'Configuration':[pretty(k) for k in ORDER]})
        for metric in selected:
            table[LABELS[metric]]=[f'{means.loc[k,metric+"_mean"]:.4f} ± {means.loc[k,metric+"_sd"]:.4f}' for k in ORDER]
        sections += ['### '+title,md_table(table,list(table.columns))]
    sections += ['## Strategy effects within each model family',
                 'All deltas below are strategy minus the no-correction configuration in the same family, paired by repeat. Changes are absolute score units, not relative percentages. A difference of 0.01 corresponds to one percentage point for recall, specificity and precision. Higher recall is not automatically better; the corresponding false-positive burden and probability error matter. All nine metrics and per-repeat/per-fold differences are in the CSV tables.']
    for model in MODELS[1:]:
        baseline=means.loc[model+'__none']
        sections += ['### '+MODEL_NAMES[model],
            f'No-correction reference: ROC-AUC {baseline.roc_auc_mean:.4f}, AP {baseline.average_precision_mean:.4f}, sensitivity {baseline.recall_sensitivity_mean:.4f}, specificity {baseline.specificity_mean:.4f}, precision {baseline.precision_mean:.4f}, F1 {baseline.f1_mean:.4f}, MCC {baseline.mcc_mean:.4f}, balanced accuracy {baseline.balanced_accuracy_mean:.4f}, Brier {baseline.brier_score_mean:.4f}.']
        part=effect_summary[effect_summary.model==model].copy()
        part['Strategy']=part.strategy.map(SHORT)
        cols=['Strategy']+['delta_'+m+'_mean' for m in ['roc_auc','average_precision','recall_sensitivity','specificity','precision','f1','mcc','balanced_accuracy','brier_score']]
        sections += [md_table(part,cols,['Strategy']+['Δ '+LABELS[m] for m in METRICS])]
        high=rw[rw.model==model].sort_values('recall_sensitivity_mean',ascending=False).iloc[0]
        sections += [f'The highest descriptive sensitivity in this family was {SHORT[high.strategy]}: {high.recall_sensitivity_mean:.4f} versus {baseline.recall_sensitivity_mean:.4f} without correction (Δ {high.recall_sensitivity_mean-baseline.recall_sensitivity_mean:+.4f}). Precision was {high.precision_mean:.4f} versus {baseline.precision_mean:.4f}, and specificity {high.specificity_mean:.4f} versus {baseline.specificity_mean:.4f}. This describes a trade-off, not superiority.']
    sections += ['## Variability: fold scores versus pooled repeat scores',
                 'The table contrasts AP variation at the two aggregation levels. Fold means/SDs summarize 25 correlated smaller folds. Repeat means/SDs summarize five overlapping, pooled repeat predictions. Smaller repeat SD does not prove H4: aggregation changes validation sample size and estimand, and the two collections are dependent. Per-repeat fold means and within-repeat fold SDs, plus all nine metrics’ min/max and between-repeat SDs, are saved separately.']
    table=pd.DataFrame({'Configuration':names,
        'Fold AP mean':[f'{means_: .4f}' for means_ in fw.average_precision_mean],
        'Fold AP SD':fw.average_precision_sd,
        'Pooled-repeat AP mean':rw.average_precision_mean,'Pooled-repeat AP SD':rw.average_precision_sd,
        'Pooled-repeat AP min':rw.average_precision_min,'Pooled-repeat AP max':rw.average_precision_max})
    sections += [md_table(table,list(table.columns)),
        'No standard errors, confidence intervals, p-values or significance tests were computed. Repeat identifiers are realizations of the fixed random-state-42 sequence; they are not independently varied seed settings.',
        '## Research questions and working hypotheses',
        'The following exact definitions were supplied by the project lead during Step 34; they were not inferred from the numerical results.',
        '**RQ2:** How do different machine-learning models perform across discrimination, minority-class detection, calibration, and robustness dimensions?',
        'The nine-metric comparisons describe discrimination, minority detection and fixed-threshold trade-offs. Brier provides probability-error evidence only. Calibration experiments are pending Step 35, so RQ2 is only partially addressed. Robustness here means variation across observed internal folds/repeats, not external generalization, subgroup fairness or independent seed robustness.',
        '**RQ3:** How do different class-imbalance strategies affect model performance and stability?',
        'Paired strategy-minus-none tables quantify changes in all nine measures and their repeat variation. Effects are model-specific: higher sensitivity must be read with precision, specificity, AP, MCC and Brier. Realized ADASYN ratios differ from the requested target and other samplers, limiting attribution to geometry alone.',
        '**RQ5:** How stable are model-performance results across repeated validation folds and random seeds?',
        'Fold/repeat SDs and ranges answer the observed repeated-validation component only. No independent random-seed sensitivity experiment was performed; that part of RQ5 remains unanswered.',
        '**H1:** Different ML models will show statistically meaningful differences across at least some discrimination, minority-class, calibration, or robustness metrics.',
        'Observed model differences are descriptive evidence motivating later comparisons. Statistical meaningfulness is untested; H1 is neither confirmed nor rejected here, and calibration remains pending.',
        '**H2:** Imbalance-handling strategy will materially change minority-class performance and may produce trade-offs between recall, precision and specificity.',
        'The measured strategy deltas describe changes in minority detection and trade-offs compatible with H2. No formal statistical or clinical materiality criterion was prespecified or tested; H2 is not statistically established.',
        '**H4:** Performance estimates from a single split will show greater apparent variability than estimates summarized across repeated stratified validation.',
        'H4 remains untested. Comparing the 25 fold SD with five pooled-repeat SDs is not an appropriate standalone single-split-versus-repeated-validation test. A separate prespecified comparison would be needed; none was performed.',
        '**H7:** Observed model differences will vary with imbalance strategy, so model ranking based on one configuration will not necessarily persist across all configurations.',
        'Descriptive within-strategy model ranks and pairwise rank reversals are saved. A rank reversal means a sign change in one model pair’s mean score difference across strategies; it is not a significance or interaction test.']
    if len(reversals):
        examples=reversals[reversals.metric.isin(['average_precision','roc_auc'])].head(6)
        sections += [md_table(examples,list(examples.columns)),f'{len(reversals)} metric/model-pair combinations had at least one descriptive rank reversal across the six strategies. These patterns are compatible with H7 but do not establish a statistically supported interaction.']
    sections += ['## Configurations worth examining later',
        'This exploratory examination set retains the highest descriptive AP and MCC configurations within each learned-model family plus each no-correction reference. It is a transparent triage rule, not a final model choice, significance claim, deployment recommendation or claim of optimal thresholds. AP and MCC target different behaviors; Brier and variability must also be reviewed.',
        md_table(shortlist[['pipeline','average_precision_mean','mcc_mean','brier_score_mean','reason_for_later_examination']],
                 ['pipeline','average_precision_mean','mcc_mean','brier_score_mean','reason_for_later_examination']),
        '## Figures',
    ]
    for figure in FIGURES:
        sections += [f'![{figure["caption"]}]({figure["file"]})',figure['caption']]
    sections += ['## Limitations and justified next steps',
        '1. Perform Step 35 probability calibration assessment on training/CV data under a leakage-safe, prespecified design, retaining no-correction references. Brier alone is not calibration evidence.',
        '2. Predefine the clinical decision objective and false-positive/false-negative costs before future threshold analysis; do not optimize thresholds on the locked held-out test set.',
        '3. After a justified examination set is agreed, plan model-specific SHAP analysis with explicit background-data provenance. Interpret explanations as associations, not causes.',
        '4. If formal H1/H7 inference is later requested, specify a paired method that accounts for repeated-CV dependence and multiplicity. Do not use naive independent-fold t-tests.',
        '5. Design a separate single-split-versus-repeated-validation comparison for H4 and independent seed experiments for the seed component of RQ5. Current data cannot supply those claims.',
        '6. Preserve held-out data for the final agreed evaluation after model-selection decisions. This analysis makes no external-validity, clinical-readiness or final-winner claim.',
        'No calibration, threshold optimization, SHAP, statistical testing, training or new experiments were performed. All results are internal descriptive CV evidence from one public dataset.',
        '## Reproduction and artifact inventory',
        'Run the saved script with the existing project Python environment. The script imports scorers only and reads only the exact saved run; it refuses input hash/coverage/metric discrepancies. Plot-only packages are isolated under `_plot_dependencies/` and do not modify `.venv` or pinned requirements. Metadata and inventory record analysis and source hashes. CSV numeric precision is retained; Markdown rounds to four decimals.',
        '```powershell\n.\\.venv\\Scripts\\python.exe results/step34_imbalance_analysis/analyze_step34.py\n```',
        'Key outputs: `comparison_fold_mean_sd.csv`, `comparison_pooled_repeat_mean_sd.csv`, `comparison_all_metrics_long.csv`, `pooled_metrics_by_repeat_with_mcc.csv`, `fold_metrics_with_mcc.csv`, `fold_mean_and_sd_within_each_repeat.csv`, `fold_vs_pooled_repeat_variability.csv`, both paired strategy-effect tables and their summary, model ranks/reversals, metric-definition diagnostics, examination set, coverage table, six 300-dpi PNGs, figure captions, `verification.json`, `analysis_manifest.json` and `artifact_inventory.csv`. Supporting dependency/configuration directories are not scientific outputs. Inventory hashes cover all deliverables except the inventory itself.',
    ]
    (OUT/'scientific_analysis_report.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    (OUT/'figure_captions.json').write_text(json.dumps(FIGURES,indent=2),encoding='utf-8')
    final_inputs={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()}
    check('all original Step33 files unchanged', inputs == final_inputs)
    check('all analysis metrics finite', np.isfinite(derived[METRICS].to_numpy()).all() and np.isfinite(pooled[METRICS].to_numpy()).all())
    verdict=dict(overall_passed=all(c['passed'] for c in CHECKS), n_checks=len(CHECKS),
                 n_passed=sum(c['passed'] for c in CHECKS), n_failed=sum(not c['passed'] for c in CHECKS), checks=CHECKS)
    (OUT/'verification.json').write_text(json.dumps(verdict,indent=2),encoding='utf-8')
    analysis_manifest=dict(source_run=RUN, source_directory=str(SOURCE.resolve()),
        source_artifact_hashes_start=inputs, source_artifact_hashes_end=final_inputs,
        analysis_script_sha256=sha(__file__), python=sys.version, platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
        analysis_scope='descriptive saved OOF/fold/repeat metrics only; no data loading, fitting, inference or model selection',
        main_aggregation='mean and sample SD of five pooled repeat scores; separate 25-fold mean and sample SD',
        random_seed_scope='existing CV random-state 42 sequence only; no independent seed experiment',
        mcc_zero_denominator_policy='sklearn convention 0, flagged separately',
        clinical_threshold=0.5, input_protocol=manifest['cv'], figures=FIGURES,
        formal_rq_hypothesis_source='Project lead supplied exact definitions in Step34 chat; report reproduces wording',
        verification={k:v for k,v in verdict.items() if k!='checks'},
        generated_at_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'analysis_manifest.json').write_text(json.dumps(analysis_manifest,indent=2),encoding='utf-8')
    inventory=[]
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in ['artifact_inventory.csv','execution.log']:
            inventory.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)))
    write_csv(pd.DataFrame(inventory),'artifact_inventory')
    print(json.dumps({'verification':analysis_manifest['verification'],'n_artifacts':len(inventory)+1,
                      'largest_saved_metric_deviation':largest_saved_deviation},indent=2),flush=True)
    print(rw[['pipeline','roc_auc_mean','average_precision_mean','recall_sensitivity_mean','specificity_mean','precision_mean','mcc_mean','brier_score_mean']].to_string(index=False),flush=True)


FIGURES=[]
if __name__=='__main__':
    main()
