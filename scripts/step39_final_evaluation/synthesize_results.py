"""Post-evaluation synthesis from saved predictions only; no test API or fitting."""
from final_common import *
import math
os.environ['MPLCONFIGDIR']=str(OUT/'_matplotlib_config');sys.path.append(str(ROOT/'results/step34_imbalance_analysis/_plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve

def table(df):
    def cell(v):
        if pd.isna(v):return 'undefined'
        return f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
    freeze=validate_seal('selection_freeze.json');deployment=validate_seal('model_and_threshold_freeze.json')
    test=pd.read_csv(OUT/'locked_test_metrics.csv',float_precision='round_trip');pred=pd.read_csv(OUT/'locked_test_predictions.csv',float_precision='round_trip')
    ci=pd.read_csv(OUT/'test_bootstrap_intervals.csv',float_precision='round_trip');exact=pd.read_csv(OUT/'exact_binomial_intervals.csv',float_precision='round_trip')
    labels=['Dummy / 0.50','LR none / 0.50','LR weighted / 0.50','RF none / 0.50','LR weighted sigmoid / 0.50','LR weighted sigmoid / 0.061']
    labelmap=dict(zip(test.id,labels));streamlabels={c['id']:labels[i].rsplit(' / ',1)[0] for i,c in enumerate(CONFIGS)}
    compact=test[['id','threshold','roc_auc','average_precision','sensitivity','specificity','precision','f1','mcc','balanced_accuracy','brier_score','log_loss','tn','fp','fn','tp']].copy();compact['configuration']=compact.id.map(labelmap)
    write(compact,'paper_ready_heldout_results')
    write(test[['id','mean_probability','mean_probability_bias','ece_quantile_5','ece_quantile_10','calibration_in_large_intercept','joint_intercept','calibration_slope','joint_fit_status','logit_clip_count']],'paper_ready_test_calibration')
    write(test[['id','tn','fp','fn','tp','precision_defined','mcc_denominator_nonzero']],'confusion_matrices')
    # Recompute development metrics only from saved CV probabilities, no training.
    original=pd.read_csv(S33/(RUN+'_oof_predictions.csv'),float_precision='round_trip');nested=pd.read_csv(S35/'nested/outer_predictions.csv',float_precision='round_trip')
    dev=[]
    for policy in deployment['policies']:
        if policy['stream']==SIG:d=nested[(nested.pipeline==BASES[2])&(nested.method=='sigmoid')].rename(columns={'y_proba':'positive_probability'})
        else:d=original[original.pipeline==policy['stream'].split('::')[0]].rename(columns={'y_proba':'positive_probability'})
        for repeat,g in d.groupby('repeat'):
            values=all_metrics(g.y_true,g.positive_probability,policy['threshold'],diagnostics=False)
            dev.append(dict(id=policy['id'],repeat=int(repeat),role='apparent_after_threshold_selection' if policy['policy']=='maximum_grid_mcc' else 'previously_viewed_CV_configuration; sigmoid_calibration_nested',**values))
    dev=pd.DataFrame(dev);write(dev,'development_metrics_by_correlated_repeat')
    metrics=BOOT_METRICS+['ece_quantile_5','ece_quantile_10','mean_probability']
    comparison=[]
    for row in test.itertuples():
        g=dev[dev.id==row.id]
        for metric in metrics:
            v=g[metric].dropna();comparison.append(dict(id=row.id,metric=metric,development_mean=float(v.mean()) if len(v) else None,development_sd_across_5_correlated_repeats=float(v.std(ddof=1)) if len(v)>1 else None,
                development_defined_repeats=len(v),development_role=g.role.iloc[0],test_value=getattr(row,metric),test_minus_development=float(getattr(row,metric)-v.mean()) if len(v) and pd.notna(getattr(row,metric)) else None,
                comparison_inference='descriptive only; different training/evaluation sample sizes; not paired datasets; no generalization-gap p-value'))
    comparison=pd.DataFrame(comparison);write(comparison,'development_versus_test')
    # Carry forward actual nested operating-policy evidence without substituting apparent maxima.
    related=pd.read_csv(ROOT/'results/step36_threshold_analysis/inner_selected_policy_summary.csv')
    related=related[(related.pipeline==BASES[2])&(related.method=='sigmoid')&(related.policy=='maximum_grid_mcc')]
    write(related,'related_step36_inner_selected_policy_evidence')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    figures=[]
    def save(name,caption):plt.savefig(OUT/name,dpi=300,bbox_inches='tight');plt.close();figures.append(dict(file=name,caption=caption))
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for stream,g in pred.groupby('stream',sort=False):
        y=g.y_true.to_numpy();p=g.positive_probability.to_numpy();fpr,tpr,_=roc_curve(y,p);pp,rr,_=precision_recall_curve(y,p)
        axes[0].plot(fpr,tpr,label=streamlabels[stream]);axes[1].step(rr,pp,where='post',label=streamlabels[stream])
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[1].axhline(50/1022,ls='--',color='gray',label='Test prevalence')
    axes[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='Frozen held-out ROC curves');axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall curves; metric reported is AP')
    for ax in axes:ax.legend(fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1.03)
    fig.tight_layout();save('figure01_test_discrimination.png','Exactly five frozen probability streams; weighted raw/sigmoid ROC and PR curves coincide because the final sigmoid map is monotone. AP is average_precision_score, not trapezoidal PR-AUC. No test threshold optimization was performed.')
    bins=pd.read_csv(OUT/'locked_test_reliability_bins.csv');bins=bins[bins.requested_bins==5]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for stream,g in bins.groupby('stream',sort=False):
        g=g[g.n>0];axes[0].errorbar(g.mean_probability,g.observed_fraction,yerr=np.vstack([g.observed_fraction-g.observed_fraction_exact95_low,g.observed_fraction_exact95_high-g.observed_fraction]),marker='o',capsize=2,label=streamlabels[stream])
    for stream,g in pred.groupby('stream',sort=False):axes[1].hist(g.positive_probability,bins=np.linspace(0,1,31),histtype='step',label=streamlabels[stream])
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='Mean predicted probability',ylabel='Observed event fraction',title='Five-quantile-bin test reliability',xlim=(0,1),ylim=(0,1));axes[1].set(xlabel='Frozen predicted probability',ylabel='Number of test patients',title='Probability distributions')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();save('figure02_test_reliability.png','Quantile bins are descriptive diagnostics, not recalibration. Whiskers are unadjusted exact binomial event-fraction intervals conditional on bin membership. Only50 events; five bins primary and ten-bin diagnostics saved. Dummy has one occupied score bin.')
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,row,label in zip(axes.flat,test.itertuples(),labels):
        matrix=np.array([[row.tn,row.fp],[row.fn,row.tp]]);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=972)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=15,color='white' if matrix[i,j]>486 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No stroke','Stroke'],yticklabels=['No stroke','Stroke'],xlabel='Predicted label',ylabel='Observed label',title=label)
    fig.suptitle('Frozen held-out decisions: 972 nonevents, 50 events');fig.tight_layout();save('figure03_confusion_matrices.png','All six frozen operating rows, including no-positive-prediction outcomes. Predicted positive means p>=frozen threshold. No rows or failures were discarded.')
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for ax,metric in zip(axes,PRIMARY):
        for j,row in enumerate(test.iloc[:5].itertuples()):
            d=comparison[(comparison.id==row.id)&(comparison.metric==metric)].iloc[0];c=ci[(ci.id==row.id)&(ci.metric==metric)].iloc[0]
            ax.errorbar(d.development_mean,j-.08,xerr=d.development_sd_across_5_correlated_repeats,fmt='o',color='#276c8e',label='CV mean ± repeat SD' if j==0 else None)
            ax.errorbar(c.estimate,j+.08,xerr=np.array([[max(0,c.estimate-c.percentile95_low)],[max(0,c.percentile95_high-c.estimate)]]),fmt='s',color='#be4b45',label='Test + bootstrap interval' if j==0 else None)
        ax.set_yticks(range(5),labels[:5]);ax.invert_yaxis();ax.set_title(metric);ax.legend(fontsize=8)
    fig.suptitle('Different uncertainty summaries: correlated-repeat SD versus fixed-model test bootstrap');fig.tight_layout();save('figure04_development_versus_test.png','CV repeat SD is not a confidence interval and repeats are dependent. Test whiskers are marginal percentile95 intervals over5000 paired patient resamples, conditional on frozen fits and excluding training/selection uncertainty. Test-minus-CV differences are descriptive.')
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for j,row in enumerate(test.itertuples()):
        for metric,ax in [('sensitivity',axes[0]),('specificity',axes[1])]:
            c=exact[(exact.id==row.id)&(exact.metric==metric)].iloc[0];ax.errorbar(c.estimate,j,xerr=np.array([[c.estimate-c.exact95_low],[c.exact95_high-c.estimate]]),fmt='o',color='#276c8e',capsize=3)
    for ax in axes[:2]:ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,1.03)
    axes[0].set_title('Sensitivity: exact95 intervals');axes[1].set_title('Specificity: exact95 intervals')
    pos=np.arange(6);axes[2].barh(pos-.16,test.fp,height=.3,label='False positives');axes[2].barh(pos+.16,test.fn,height=.3,label='False negatives');axes[2].set_yticks(pos,labels);axes[2].invert_yaxis();axes[2].set_title('Actual error counts');axes[2].legend()
    fig.tight_layout();save('figure05_detection_tradeoffs.png','Exact Clopper-Pearson marginal intervals condition on50 events and972 nonevents. Precision intervals and undefined PPV flags are saved separately. No multiplicity-adjusted claim or cost-based clinical utility is inferred.')
    dump(figures,'figure_captions.json')
    definitions={
        'RQ1':('Can machine-learning models provide robust, well-calibrated, and interpretable stroke-risk predictions under severe class imbalance?','Partial evidence within one dataset: discrimination above the dummy, model-dependent reliability, substantial detection/FP trade-offs and interpretable fitted behavior. Broad robustness, external calibration and clinical utility not established.'),
        'RQ2':('How do different machine-learning models perform across discrimination, minority-class detection, calibration, and robustness dimensions?','Steps33–35 and frozen test show different probability/detection profiles; no universal winner or superiority test on final test.'),
        'RQ3':('How do different class-imbalance strategies affect model performance and stability?','Steps33–34 compare25 configurations; weighting/undersampling increase recall with specificity/precision/Brier trade-offs. Only weighting has a final test representative here; no new test-guided sampler comparisons.'),
        'RQ4':('Does high discriminative performance necessarily correspond to well-calibrated predicted probabilities?','Weighted LR raw/sigmoid share final AUC/AP but materially different Brier and probability distributions. Brier is not calibration alone; small test calibration diagnostics remain uncertain.'),
        'RQ5':('How stable are model-performance results across repeated validation folds and random seeds?','Partition stability under seed42 described; approximate corrected-CV inference and conditional bootstrap limitations explicit. Independent model/CV seed sensitivity untested.'),
        'RQ6':('Which features contribute most strongly to stroke-risk predictions according to SHAP-based analysis?','Step38 exact finite-background original-feature SHAP: age dominant, lower ranks vary; in-sample model behavior only, no causal/generalization evidence.'),
        'RQ7':('How does changing the classification threshold affect sensitivity, specificity, precision, false positives and false negatives?','Step36 development policies and two frozen sigmoid-LR test cutoffs quantify trade-offs without test optimization.'),
        'RQ8':('Are observed performance differences between models and experimental configurations statistically consistent?','Step37 limited exploratory corrected repeated-CV tests/Holm54 slots:19 flags, none for AP/AUC; no final-test hypothesis family or significance claims added.'),
        'H1':('Different ML models will show statistically meaningful differences across at least some discrimination, minority-class, calibration, or robustness metrics.','Limited approximate exploratory evidence only; not confirmatory global significance, equivalence or universal superiority.'),
        'H2':('Imbalance-handling strategy will materially change minority-class performance and may produce trade-offs between recall, precision and specificity.','Observed trade-offs quantified in CV and frozen weighting examples; clinical materiality requires costs/margins, not supplied.'),
        'H3':('A model/configuration with strong ROC-AUC will not necessarily have the best probability calibration.','Descriptively compatible: identical monotone-map ranking with contrasting Brier/reliability. No standalone calibration-superiority hypothesis test.'),
        'H4':('Performance estimates from a single split will show greater apparent variability than estimates summarized across repeated stratified validation.','Untested. One held-out split versus repeated CV has different sample sizes, fitted models and estimands; comparing their SD/intervals does not test H4.'),
        'H5':('Changing the classification threshold will produce meaningful sensitivity-specificity and false-positive/false-negative trade-offs.','Descriptively demonstrated by frozen cutoffs; meaningful clinical utility remains unestablished.'),
        'H6':('Important predictors identified by SHAP will be model-dependent rather than identical across every classifier.','Descriptive compatibility from Step38 ranks/shares; different output scales/encodings and reference samples limit mechanistic interpretation; no formal test.'),
        'H7':('Observed model differences will vary with imbalance strategy, so model ranking based on one configuration will not necessarily persist across all configurations.','Step37 weighting/Brier interaction has approximate Holm-adjusted exploratory evidence, plus strategy-dependent ranks; no global interaction claim.'),
        'H8':('Explicit dataset-quality and preprocessing checks will identify characteristics that materially affect interpretation or reproducibility of results.','Audit evidence available:5110 rows,249 events,201 BMI missing,1544 Unknown smoking, no duplicate records; median imputation, explicit Unknown category, ID exclusion, source hashes and fold-local transformations matter to reproducibility. No randomized causal test of audit benefits; external validity not established.')}
    matrix=pd.DataFrame([dict(item=k,official_definition=v[0],evidence_and_status=v[1],scope='single public dataset; descriptive/exploratory unless explicitly untested') for k,v in definitions.items()]);write(matrix,'research_question_hypothesis_evidence_matrix')
    audit=pd.DataFrame([dict(issue=x,status='disclosed; no test-integrity blocker') for x in freeze['known_discrepancies']]+[
        dict(issue='No held-out test data were loaded until both freeze records and independent pre-access validation existed.',status='verified'),
        dict(issue='Step38 preferred representatives are exploratory and in-sample SHAP was not used for generalization-based selection.',status='verified'),
        dict(issue='Final MCC cutoff uses existing nested-CV sigmoid probabilities; its training maximum is apparent, not unbiased selected-policy performance.',status='disclosed'),
        dict(issue='Default CSV parsing changed near-tie exact-equality replay before test access; round-trip parsing recovered the frozen numerical threshold without changing any freeze.',status='resolved before access; recorded')]);write(audit,'evidence_audit_and_manuscript_corrections')
    chosen=test.iloc[-1];sci=exact[(exact.id==chosen.id)&(exact.metric=='sensitivity')].iloc[0]
    uncertainty=ci[(ci.id==chosen.id)&ci.metric.isin(PRIMARY)]
    freeze_time=freeze['created_utc'];deployment_time=deployment['created_utc'];access=json.loads((OUT/'test_access_started.json').read_text())['created_utc']
    results_text=f'''# Paper-ready Results

The development experiment evaluated five baseline families and25 supported model/imbalance configurations using5-fold ×5-repeat stratified validation on4,088 patients (199 events), generating625 successful fold fits and511,000 repeated OOF predictions. Each repeat predicted each development patient once; repeated predictions are not independent observations. No boosting models were evaluated.

A limited exploratory final comparison was frozen before held-out access rather than declaring a universal winner. Four base pipelines and one sigmoid map produced five probability streams and six predefined operating rows. Calibration used verified five-fold OOF weighted-LR decision scores; the final weighted base was refitted on all development patients. The extra cutoff0.061 was selected using the highest threshold within1e-12 of maximum mean MCC over five separate nested-sigmoid OOF repeats and the established191-point grid.

The locked test contained1,022 patients:50 events and972 nonevents. The final numerical results are:

{table(compact[['configuration','roc_auc','average_precision','sensitivity','specificity','precision','f1','mcc','balanced_accuracy','brier_score','log_loss','fp','fn']])}

At the frozen0.061 cutoff, sigmoid-calibrated weighted LR detected40/50 events (sensitivity0.80; exact95% interval[{sci.exact95_low:.4f}, {sci.exact95_high:.4f}]), with216 false positives, specificity0.7778 and precision0.15625. Its MCC was0.28764. The same calibrated probabilities at0.50 produced no positive predictions. Uncorrected LR at0.50 detected just one event with no false positives; its precision of1.0 is based on only one predicted positive and must not be interpreted as reliable high PPV. Dummy, RF and sigmoid-weighted LR at0.50 had undefined PPV and zero MCC by convention because they predicted no positives.

Raw and sigmoid-weighted LR had identical final AP/AUC because one monotone map was applied to the same final base scores, while Brier changed from0.167627 to0.040734. This is a probability-scale change, not a discrimination gain. Uncorrected LR had AP0.270010, ROC-AUC0.842119 and Brier0.040636; RF had AP0.213282, ROC-AUC0.825165 and Brier0.042266. No test significance comparison or new winner was selected.

Marginal percentile intervals from5,000 paired test-patient resamples and exact binomial detection/PPV intervals are provided separately. Resampling preserves the pairing of all model predictions for each test patient, with no model refits. PPV bootstrap intervals are unavailable if any draw has no positive predictions; no undefined draws are silently discarded.

For the0.061 operating row, the fixed-model bootstrap discrimination/probability intervals are:

{table(uncertainty[['metric','estimate','percentile95_low','percentile95_high']])}
'''
    discussion='''# Paper-ready Discussion

The evidence reinforces the need to separate ranking performance, probability accuracy and operating-point utility under severe imbalance. Weighting increased detection at the established0.50 cutoff, with many additional false positives and poorer Brier score. Sigmoid calibration improved weighted-LR probability accuracy while preserving the ranking of the final fixed model; at0.50 it substantially reduced positive classifications. A training-selected lower cutoff restored detection but retained low precision. These findings do not establish a clinically optimal threshold without explicit costs, benefits, resource constraints and external validation.

The held-out AP values differ from development means, but there are only50 held-out events and no valid paired generalization-gap test between these different evaluation samples. The reported test intervals condition on fixed trained pipelines/selection choices and assume exchangeable patients from a comparable source population. They omit training, configuration-selection and threshold-selection uncertainty, and do not establish transportability. Bootstrap degeneracy at no-positive endpoints does not prove zero uncertainty: exact binomial intervals for sensitivity/specificity are preferred. The many reported intervals are marginal, not simultaneous confidence statements.

Repeated-CV overlap was explicitly addressed in Step37 using approximate corrected repeated-CV tests with a54-slot exploratory Holm family. Nineteen adjusted flags arose under that approximation, with no AP/AUC comparison surviving correction. Conditional patient bootstrap intervals in Step37 describe saved-score patient-mix sensitivity and exclude shared-training/refitting uncertainty. Neither approach provides exact full-process variance guarantees. H4 and independently designed multi-seed sensitivity remain untested; one final test split does not repair those gaps.

The custom categorical-aware ADASYN standardizes continuous features within each training fold, operates in mixed standardized/one-hot neighbor geometry, inverse-transforms continuous synthetic values and decodes valid categories. It is not stock ADASYN. Its achieved ratios ranged0.506911–0.522662 despite requested0.5 because of per-seed rounding, a documented comparison caveat. No synthetic patient was used for the real-patient SHAP illustrations, and this final limited test comparison adds no sampler experiments.

Step38 found age dominant across representative fitted models, with model-dependent lower ranks. These in-sample, finite-background SHAP explanations describe model behavior on log-odds or probability scales; they do not identify causes or demonstrate out-of-sample explanatory stability. Windows blocked the SHAP/Numba library, so exact analytical linear SHAP and exact original-feature coalition enumeration were independently verified instead. Categorical groups were masked together; correlated clinical-feature hybrids remain a limitation.

The public dataset has249 events among5,110 observations,201 missing BMI entries and1,544 Unknown smoking-status entries, with no duplicate records in the prior audit. These features motivate fold-local median imputation, explicit Unknown encoding, leakage safeguards and attention to population composition. Results concern the recorded stroke label on one public dataset, not prospective incidence or a validated risk horizon. Calibration and clinical usefulness require external, preferably temporal/geographic validation, assessment of subgroup performance and a justified deployment population. No clinical deployment readiness is claimed.
'''
    (OUT/'paper_ready_results.md').write_text(results_text,encoding='utf-8');(OUT/'paper_ready_discussion.md').write_text(discussion,encoding='utf-8')
    tasks='''# Remaining manuscript and reference tasks

* Remove XGBoost, LightGBM and CatBoost from evaluated-model/result claims; five evaluated baseline families were Dummy, LR, Decision Tree, RF and RBF SVM. Retain boosting only as proposed future work if appropriate.
* Identify the source dataset, license/version and original provenance in the bibliography; do not infer a prospective prediction horizon or representative clinical population.
* Cite primary sources for AP/PR under imbalance, Brier/probability calibration, Platt sigmoid calibration, corrected repeated-CV inference, Holm correction, SHAP, SMOTENC and ADASYN. Explain that the project's custom sampler is not a published stock implementation.
* Verify every bibliographic record/DOI against the writing team's library; this package does not fabricate references or claim a complete literature review. Primary implementation references used here: https://scikit-learn.org/stable/modules/calibration and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html . Step37/38 reports contain their method-source links.
* Revise claims about H4, independent seeds, statistical significance, calibration versus discrimination, causal explanation and clinical readiness using the evidence matrix. RQ1 and H8 use the exact official wording supplied by the research lead.
* Add limitations for exploratory configuration choice after CV inspection, fifty test events, dependent repeats, fixed-model intervals, threshold development optimism, one public dataset and in-sample SHAP.
* Manuscript PDFs/DOCX files were not provided in the repository for line-by-line editing. Apply these claim corrections to the actual draft and verify reference completeness before submission.
* The final test is now spent. Do not optimize thresholds, recalibrate, select another model or initiate new test-guided comparisons. Any future protocol needs new independent evaluation data; no such experiment was started here.
'''
    (OUT/'remaining_manuscript_tasks.md').write_text(tasks,encoding='utf-8')
    report=['# StrokeSure Step39: final locked-test evaluation and synthesis',
        '## Pre-test audit, selection and chronology',
        f'Protocol freeze: {freeze_time}; deployment/model/threshold freeze: {deployment_time}; first test-access marker: {access}. Both JSON records were exclusive-created, made read-only and SHA-256 sealed; independent pre-access validation passed. They provide an application-level audit seal, not adversarial immutable storage. The protocol hash is `{sha(OUT/"selection_freeze.json")}` and deployment hash `{sha(OUT/"model_and_threshold_freeze.json")}`.',
        'The comparison is exploratory and limited; no universal winner was invented. Exact configurations are Dummy none, LR none, LR balanced class-weight, RF none, and sigmoid weighted LR. All use threshold0.50; sigmoid weighted LR additionally uses training-selected0.061. No isotonic, resampling or SVC final comparison was added. No Step38 in-sample SHAP evidence was treated as generalization evidence or the primary selection criterion. The no-correction references, weighting trade-off and sigmoid probability repair jointly cover discrimination, minority detection, reliability and observed partition stability.',
        'Calibration reused4088 Step35 repeat1 OOF decision scores from five established folds, each produced by a model not fitted on that observation. One Platt-smoothed sigmoid map was fitted to those scores/labels; four base pipelines were fitted on all4088 training patients. This corresponds to the [OOF/single-refit calibration pattern](https://scikit-learn.org/stable/modules/calibration), not averaging an ensemble of five final models. Full-training score-distribution versus fold-model score-distribution mismatch remains a limitation. Exact coefficients, score IDs, model hashes and fit parameters are retained.',
        'The additional threshold uses all five saved nested-calibration OOF repeats, evaluates the fixed191-point grid separately within each repeat, averages correlated-repeat MCC, and takes the largest threshold within1e-12 of the maximum. This is training/development selection, not independent patient replication. Its selected development maximum is optimistic. Step36 related inner-selected policy evidence is supplied separately and is not relabeled as unbiased performance of this exact final selector.',
        table(audit),results_text,discussion,
        '## Probability reliability and uncertainty',table(test[['id','mean_probability','ece_quantile_5','ece_quantile_10','calibration_in_large_intercept','calibration_slope','joint_fit_status']]),
        'Five-quantile-bin reliability is primary; ten-bin ECE is sensitivity only. Brier measures overall probability accuracy, not calibration alone. Intercept/slope are descriptive test-prediction diagnostic fits and are never applied as new calibrators. Their optimizer status and logit clipping are retained; constant scores have no estimable joint slope. No post-test fitting of a predictive model or deployed calibration map occurred.',
        'Exact95% Clopper-Pearson sensitivity/specificity/PPV intervals use conditional binomial denominators ([SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)). Ordinary paired patient bootstrap uses5000 shared multinomial count vectors, seed39042, including all draws. Intervals are fixed-model marginal/unadjusted, not training-process or simultaneous intervals. PPV with no predicted positives is undefined; MCC zero-denominator uses the explicit zero convention. No p-values, rankings chosen by test results or bootstrap significance declarations were added.',
        '## Development versus held-out evidence',
        'development_versus_test.csv reports mean and SD across five dependent pooled repeats, separate from test point estimates. SD is not a confidence interval. Fixed0.50 configuration estimates are previously inspected CV evidence; sigmoid calibration fitting was nested but the configuration choice was not. The0.061 development row is explicitly apparent after threshold selection and must not be quoted as an unbiased selected-policy estimate. Test-minus-development values are descriptive and never used for re-selection.',
        '## Research-question and hypothesis mapping',table(matrix),tasks,
        '## Verification and integrity',
        'pretest_verification.json has267/267 passing checks; independent_preaccess_validation.json certifies freeze/threshold replay before access. test_execution_verification.json has6/6 passing checks. The independent delivery verifier replays metrics from saved test probabilities, confusion counts, bootstrap arithmetic and freeze chronology without reloading the raw test data or re-predicting models. It checks all original artifacts/source hashes. Final inventory and verification files index this package.',
        '## Figures']
    for f in figures:report += [f'![{f["caption"]}]({f["file"]})',f['caption']]
    report += ['## Reproduction boundaries','Training/selection and test scripts are one-shot and refuse repeat access. Do not rerun them against a spent test. Synthesis and independent verification can be rerun from saved predictions only: `.\\.venv\\Scripts\\python.exe results/step39_final_evaluation/synthesize_results.py` and `verify_delivery.py`. Hash/inventory finalization is separate so active logs are hashed after closure. No commit/push was made.']
    text='\n\n'.join(report)+'\n'
    # Editorial readability only, never changes frozen numerical code or predictions.
    for a,b in {'Step39':'Step 39','Step36':'Step 36','Step37':'Step 37','Step38':'Step 38','Steps33':'Steps 33','among5':'among 5','on4,':'on 4,','contained1,':'contained 1,','patients:50':'patients: 50','and972':'and 972','only50':'only 50','of1.0':'of 1.0','seed42':'seed 42','uses5000':'uses 5,000','seed39042':'seed 39042','cutoff0.':'cutoff 0.','at0.':'at 0.','threshold0.':'threshold 0.','selected0.':'selected 0.','reused4088':'reused 4,088','all4088':'all 4,088'}.items():text=text.replace(a,b)
    (OUT/'final_scientific_results_report.md').write_text(text,encoding='utf-8')
    assert prior_snapshot()==freeze['original_artifact_hashes']
    dump(dict(created_utc=now(),source_run=RUN,source_freeze_sha256=sha(OUT/'selection_freeze.json'),deployment_freeze_sha256=sha(OUT/'model_and_threshold_freeze.json'),heldout_evaluation_complete=True,model_fits=4,calibrator_fits=1,reused_oof_base_fits=5,test_rows=1022,test_events=50,probability_streams=5,operating_rows=6,code_hashes={p.name:sha(p) for p in OUT.glob('*.py')},plotting_version=matplotlib.__version__,original_artifacts_unchanged=True),'analysis_manifest.json')
    print('SYNTHESIS COMPLETE; no model fitting or test API access',flush=True)
if __name__=='__main__':main()
