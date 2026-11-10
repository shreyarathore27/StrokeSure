# Beyond Accuracy: A Robust and Explainable Machine Learning Framework for Stroke Risk Prediction Under Class Imbalance

**Shreya Rathore; Vaibhavi Singh**

Department of Computer Science and Engineering (Artificial Intelligence), Indira Gandhi Delhi Technical University for Women, Delhi, India

## Abstract

Severe class imbalance makes high classification accuracy an inadequate description of stroke-prediction performance. We completed a reproducible evaluation integrating discrimination, minority-class detection, probability reliability, repeated validation, statistical comparison, explainability and threshold behavior. The public Stroke Prediction Dataset contained 5,110 observations and 249 stroke-positive records. A stratified split reserved 1,022 observations, including 50 events, for final evaluation. On the 4,088-observation development partition, five baseline families and six imbalance strategies yielded 25 supported configurations, 625 fold fits and 511,000 predictions under five-fold, five-repeat cross-validation. Calibration was evaluated with nested training-only procedures. A limited representative comparison and its operating rules were sealed before one-shot test access. Uncorrected Logistic Regression (LR) achieved test ROC-AUC 0.8421, Average Precision (AP) 0.2700 and sensitivity 0.0200 at 0.50. Weighted LR achieved sensitivity 0.8000 but specificity 0.7428 and Brier 0.16763. Sigmoid calibration preserved its ROC-AUC and AP while reducing Brier to 0.04073. At a training-selected frozen cutoff of 0.061, the calibrated model detected 40 of 50 events with 216 false positives and 10 false negatives. Exploratory dependence-corrected comparisons provided no multiplicity-adjusted AP or ROC-AUC finding. Exact finite-background SHAP identified age as dominant but lower feature ranks differed across representatives. The contribution is integrated empirical accounting, not a new clinical predictor or universal model winner. Limited events, exploratory selection, correlated validation and absence of external validation constrain generalization.

**Index Terms:** stroke prediction; class imbalance; Average Precision; calibration; repeated validation; SHAP; threshold analysis; reproducibility.

## I. Introduction

Stroke prediction is a consequential application of statistical learning, but a model's ability to order observations by score is distinct from its ability to detect minority events, estimate probabilities or support a particular decision. This distinction is especially important when the recorded positive class is uncommon. The present study concerns classification of the dataset's stroke label, not prospective incidence at a validated future risk horizon. Its title preserves the original research framing; predicted scores should not be treated as established individual clinical risks.

On the audited data, a majority-only rule could achieve approximately 95% accuracy without detecting a single stroke-positive observation. ROC-AUC summarizes ranking across thresholds but does not describe the false-positive burden at a chosen cutoff. AP adds precision-recall information under imbalance [3], while MCC and balanced accuracy complement class-specific rates [4]. None substitutes for a decision analysis with explicit clinical costs.

Probability reliability is another dimension. A classifier can retain its ranking while its probability scale changes substantially. Calibration, Brier score and log loss therefore require separate attention [8], [9]. Likewise, changing a threshold can alter sensitivity and specificity without improving ranking or calibration. An apparent improvement across two operating points cannot automatically be attributed to the calibrator if their cutoffs also differ.

The study completes the original proposal's integrated evaluation: model-by-strategy benchmarking, fold-local processing, repeated validation, nested calibration, threshold policies, qualified statistical comparisons and global/local explanations. It adds a frozen, limited held-out comparison rather than selecting a winner from the test results. The contribution lies in transparent connections among these dimensions and auditable artifacts. We do not claim the dataset is novel, the component methods are new, or this is the first framework to combine them. Independent seed sensitivity and a direct single-split-versus-repeated-estimation variability experiment were not performed.

Established stroke-risk factors include increasing age, hypertension, cardiac disease, diabetes or raised glucose, excess weight and tobacco exposure [32], [33]. The dataset contains related variables, but its broad heart-disease flag does not specifically identify atrial fibrillation, average glucose is not a confirmed fasting measurement or diabetes diagnosis, and BMI has different interpretations across ages. These fields are imperfect proxies for clinical constructs. Their associations and model attributions cannot establish causal effects or a validated prospective risk estimate.

## II. Related Work

### A. Dataset-specific predictive comparisons

Dritsas and Trigka compared learners and stacking [2]. Tazin et al. compared LR, DT, RF and voting with SMOTE [16], while Al Duhayyim et al. studied a voting ensemble combining RF, XGBoost and LightGBM [17]. Sahriar et al. coupled projection-based feature extraction with classification and statistical feature analysis [18]. Akinwumi et al. compared five learners with random oversampling and five-fold validation [20]. These studies establish substantial prior multi-model and imbalance work; their reported scores cannot be pooled with ours because cohorts, resampling order, optimization and validation differ.

Alruily et al. used SMOTE, a tuned ensemble, ten-fold CV and varying split random states [23]. Ivanov et al. examined optimization, missingness and sampling alternatives [24]. Kitova et al. compared five classifiers with ten-fold CV, dropping missing-BMI records rather than imputing them [27]. Shobayo et al. compared RF with LR and DT, using a training/test/validation partition [29]; its RF-focused title does not imply a single-model experiment. Thus neither CV nor random-state variation is a new contribution here.

### B. Explainability and integrated probability/decision frameworks

Chakraborty et al. combined feature selection with game-theoretic EnShap model ranking [19]; this should not be conflated with our original-feature patient-level additive explanations. Dubey et al. applied LIME and SHAP to optimized boosting [25]. Srinivasu et al. considered multiple sampling approaches with LIME and permutation importance [28]. These overlap directly with imbalance and interpretation components, although their exact validation and source-cohort alignment must not be inferred from abbreviated summaries.

Mustapha et al.'s GSM already integrates GAM, SHAP, calibration diagnostics, a calibrated score and a decision layer [21]. Its five-fold comparisons include discrimination and probability error, and its illustrative MCDA layer incorporates synthetic contextual inputs. Pranil et al.'s hybrid Bayesian-network study combines a factorial design including SMOTE, calibration assessment, statistical modelling, repeated stratified resampling and a validation-selected threshold [22]. These are closely related integrated frameworks, rather than evidence that previous research addressed only isolated techniques. Their specific architectures and validation choices differ from StrokeSure's fixed model-by-strategy benchmark; neither framework's existence establishes clinical utility for ours.

### C. Literature scope, source identity and integrity

The focused verification covers the fifteen supplied entries, not a systematic search proving an unoccupied research gap. An explicit fedesoriano citation is evidence of reported source use, not proof of byte-identical analytic cohorts: Dubey et al. report a smaller working cohort [25], and Kitova et al. use complete cases [27]. Chakraborty et al.'s data-availability link conflicts with its raw-size description [19]. Wu et al. describe a compatible dataset but cite a different uploader link [26]. Exact creator-specific source identity was not established from accessible text for Al Duhayyim et al. [17]. Srinivasu et al. explicitly cite the fedesoriano source [28]. These uncertainties are preserved in the accompanying literature matrix instead of counting all fifteen as confirmed identical-dataset studies. Mochurad et al. reported partial use alongside a separate synthetic dataset, but the article was subsequently retracted [30], [31] and is not supporting performance evidence.

### D. Evaluation foundations

Saito and Rehmsmeier explain the value of precision-recall analysis under imbalance [3]; Chicco and Jurman discuss limitations of accuracy and F1 and the complementary information in MCC [4]. These motivate a metric portfolio rather than a single optimization target. We specifically report sklearn Average Precision, not an interchangeable trapezoidal PR-AUC.

SMOTE interpolates minority examples [5], while ADASYN allocates more synthetic observations to difficult neighborhoods [6]. Mixed numerical/categorical data require suitable representations. We used the imbalanced-learn toolkit [7] and its SMOTENC implementation [15], alongside a project-specific categorical-aware ADASYN. The latter changes the representation and decoding around adaptive synthesis; results must not be attributed to unchanged stock ADASYN.

Niculescu-Mizil and Caruana compared probability behavior and post-hoc calibration [8]. Van Calster et al. emphasize evaluation of calibration in addition to discrimination and the sample-size demands of reliability assessment [9]. Our nested calibration component operationalizes that distinction without assuming every model benefits from recalibration.

Repeated validation creates correlated estimates because observations and training sets recur. Nadeau and Bengio [10] and Bouckaert and Frank [11] motivate approximate variance corrections, whereas Bengio and Grandvalet show that a universal unbiased variance estimator for ordinary K-fold CV does not exist [12]. Holm's procedure addresses multiplicity conditional on valid component tests [13]; it cannot repair an inappropriate variance model. SHAP provides an additive framework for model explanations [14], but attribution depends on the model output and the meaning of missing features. Neither interventional masking nor explanatory agreement establishes causality.

## III. Research Gap and Research Questions

### A. Research gap

The research need addressed here is a transparent common-protocol comparison of the implemented model families and imbalance strategies, with linked discrimination, minority detection, probability error, partition variability, statistical uncertainty, explanations, threshold behavior and quality audits. GSM and the hybrid Bayesian-network framework already integrate closely related dimensions [21], [22]. Accordingly, the contribution is the particular paired benchmark, documented categorical sampler, nested probability/decision safeguards and artifact-level accounting, rather than first use of any component or first integrated stroke-prediction framework. This focused review does not establish that no previous study has used an identical combination. Model comparisons were kept paired on common folds, and configuration-selection optimism was distinguished from calibration nesting and final test separation.

### B. Research questions

The eight questions from the original manuscript are retained verbatim:

**RQ1:** Can machine-learning models provide robust, well-calibrated, and interpretable stroke-risk predictions under severe class imbalance?

**RQ2:** How do different machine-learning models perform across discrimination, minority-class detection, calibration, and robustness dimensions?

**RQ3:** How do different class-imbalance strategies affect model performance and stability?

**RQ4:** Does high discriminative performance necessarily correspond to well-calibrated predicted probabilities?

**RQ5:** How stable are model-performance results across repeated validation folds and random seeds?

**RQ6:** Which features contribute most strongly to stroke-risk predictions according to SHAP-based analysis?

**RQ7:** How does changing the classification threshold affect sensitivity, specificity, precision, false positives, and false negatives?

**RQ8:** Are observed performance differences between models and experimental configurations statistically consistent?

Results below address these questions with different strengths of evidence. RQ5 is answered only for observed fold/repeat stability under one model/CV seed. RQ8 receives assumption-dependent exploratory inference, not confirmatory proof. The working hypotheses H1-H8 are mapped in Section IX and in the accompanying complete evidence matrix; H4 remains untested.

## IV. Dataset and Data Quality Analysis

### A. Dataset and predictors

We used the fedesoriano Stroke Prediction Dataset [1]. The original audit, rather than a new inspection of test observations, supplies Table I. The ten predictors were age, average glucose, BMI, hypertension, heart disease, gender, ever-married status, work type, residence type and smoking status. The identifier and target were excluded from predictors. There were three continuous, two binary and five categorical predictors. Categories include pediatric work/age patterns; the population is not an adult-only clinical cohort.

**TABLE I. Audited dataset and fixed partitions. Counts are observations unless a feature count is specified.**

| Characteristic | Value |
| --- | --- |
| Observations / raw columns | 5,110 / 12 |
| Stroke / non-stroke | 249 / 4,861 |
| Missing BMI / Unknown smoking | 201 / 1,544 |
| Duplicate records | 0 |
| Training observations / events | 4,088 / 199 |
| Test observations / events | 1,022 / 50 |
| Original predictors / transformed LR-SVM / tree | 10 / 16 / 21 |

### B. Quality findings and interpretation

The audit identified missing BMI and a large explicit Unknown smoking category, but no duplicate records. Unknown was retained as a category, not converted into an imputed smoking behavior. Median BMI imputation preserved sample size; it cannot remove bias from informative missingness. Extreme values were investigated without automatic outcome-guided removal. A unique identifier is not evidence of clinical independence, representative sampling or valid outcome ascertainment.

The dataset card supplies limited clinical provenance. A clearly established prospective risk horizon, recruitment frame and external clinical representativeness were not available for this study. The label, sampling process and mixed-age composition constrain interpretation. Dataset-wide exploratory summaries were produced before the development/test partition was established. These historical summaries limit a strong claim of complete test blindness, even though later fitting, calibration and decision selection were confined to development data and final API access was sealed. This manuscript reports that chronology rather than retrospectively describing the whole workflow as fully blinded.

A specific research-integrity example requires proportionate interpretation. Mochurad et al.'s article [30] was retracted on June 15, 2026 [31]. The notice describes unresolved dataset provenance/reliability and ethical/data-governance concerns, including inability to confirm clinical diagnosis validation; editors stated that additional provenance and accuracy information was unavailable. It concerns the reported work and its sources. It does not establish that the fedesoriano dataset is invalid, identify a defect in StrokeSure's computations, or make basic quality checks a substitute for clinical validation. We cite the retracted article only to identify that history, not to substantiate predictive performance.

## V. Methodology

### A. Split and fold-local preprocessing

An 80/20 stratified split with random state 42 produced the fixed development and test partitions. The held-out partition was not used for fitting or tuning in the completed model experiments. Five-fold stratified CV was repeated five times on development data with the established seed 42 sequence. Every configuration used the same folds. Each repeat predicted each development observation once; the five predictions per patient were not independent observations.

Continuous variables underwent median imputation. LR/SVM pipelines additionally standardized them and used drop-first one-hot categorical encoding, yielding 16 transformed predictors. Tree-family pipelines used unscaled imputed continuous values and full one-hot encoding, yielding 21. Binary predictors passed through. Category orders were fixed, Unknown smoking retained an explicit indicator, and unseen categories were tolerated. Imputation, scaling, encoding and sampling were fitted inside each training fold; validation rows were only transformed. This corrects the original schematic's ambiguous placement of preprocessing before splitting.

### B. Evaluated models

The evaluated families were a prior-probability Dummy, LR, Decision Tree (DT), Random Forest (RF) and RBF SVM. LR used L2 regularization with C=1, lbfgs and max_iter=1000. DT used Gini impurity, max_depth=5 and min_samples_leaf=20. RF used 200 trees, Gini impurity, no fixed maximum depth, min_samples_leaf=5, sqrt feature sampling and bootstrap aggregation. SVM used C=1, RBF kernel, gamma=scale and probability=True. Its existing internal Platt probability estimation was retained and is distinct from the additional nested calibration component. Random state 42 and the original parameter settings were preserved; no hyperparameter search was performed. XGBoost, LightGBM and CatBoost were proposal candidates, not evaluated models.

### C. Imbalance strategies and synthetic geometry

The six strategies were none, balanced class weighting, random oversampling (ROS), random undersampling (RUS), SMOTENC and custom CategoricalADASYN. Dummy supported none only; the four learned families supported all six, giving 25 configurations. Sampling requested a minority-to-majority ratio of 0.5 with five neighbors and seed 42. Class weighting changed the fit objective without generating rows.

For resampling, a fold-fitted sampling encoder imputed continuous variables and coded discrete variables; a decoder then restored valid categories before the established model preprocessor. SMOTENC treated nominal and continuous features separately [15]; its target audit distinguished the number of minority rows to generate from the final minority count. The sampler was never applied to validation or test observations.

Custom CategoricalADASYN fitted StandardScaler only to current-fold continuous training features, concatenated their standardized values with unit one-hot categorical blocks, and ran adaptive synthesis in that mixed representation. Continuous synthetic values were inverse-transformed to original units. Each categorical block was decoded by argmax and mapped to the actual observed category code, including noncontiguous code sets. This reconstruction preserves categorical validity but is a custom modeling assumption. Per-seed adaptive allocation and rounding produced achieved ratios of 0.506911-0.522662 rather than exactly 0.5. Actual counts and ratios were retained per fold; these deviations were not hidden or automatically treated as failures. Neither discrete decoding nor valid category codes proves that synthetic clinical combinations are realistic.

### D. Metrics and aggregation

Discrimination was measured with ROC-AUC and AP. Classification metrics at p>=cutoff were sensitivity, specificity, precision, F1, MCC and balanced accuracy. Brier score and log loss measured probability accuracy; reliability bins, expected calibration error (ECE), calibration-in-the-large and diagnostic intercept/slope complemented them. Brier is not a pure calibration statistic. Five-quantile-bin test reliability was primary; ten-bin diagnostics assessed binning sensitivity. Diagnostic calibration coefficients were never applied to test predictions.

Fold mean/SD and pooled-within-repeat metrics were kept distinct. AP is nonlinear, so pooling validation predictions within a repeat need not reproduce the mean fold AP. Table II uses the mean of five pooled-repeat metrics; their SD describes dependent repeats, not a confidence interval. Original CV tables used the documented zero convention for precision with no positive predictions. Final test tables instead mark such PPV as undefined and retain the explicit MCC zero-denominator convention.

### E. Nested probability calibration

Six development configurations were selected exploratorily: LR none, weighted LR, LR-RUS, LR-CategoricalADASYN, RF none and SVM-RUS. For each outer training fold, five inner fits generated natural-prevalence OOF calibration scores. Sigmoid with Platt target smoothing and monotone isotonic regression were fitted on those inner scores/labels. The base pipeline was refitted on the outer training data, and the uncalibrated reference and two maps were evaluated on disjoint outer validation rows. Base fits were shared across maps. LR/SVM used decision scores; RF used positive-class probability. Configuration selection was based on previously examined development results and was not nested, even though calibrator fitting/evaluation was nested.

### F. Threshold policies

The established grid had 191 cutoffs: zero, increments of 0.001 through 0.100, then increments of 0.01 through 1.00. Descriptive sweeps covered 37 unique probability streams. The protected inner-selected policy evaluation covered 17 streams with available inner probabilities. Policies included fixed 0.50, a development sensitivity constraint of at least 0.80, specificity of at least 0.90, and maximum grid MCC. Maximum-MCC ties within absolute tolerance 1e-12 chose the highest cutoff. Constraints were hypothetical operating rules, not clinical recommendations, and could fail on outer validation. Apparent optima selected on evaluated OOF outcomes were explicitly separated from inner-selected outer-fold evaluations. Original SVM probability-policy reconstruction was unavailable where inner Platt probabilities had not been saved.

### G. Dependence-aware exploratory comparisons

Eight paired configuration contrasts and two LR-versus-RF strategy interactions were specified for the statistical analysis after earlier results had been viewed. For 25 paired fold-score differences with sample variance s_d squared, the corrected standard error was sqrt[(1/25+q) s_d squared], where q was the mean original validation/training size ratio, approximately 0.25. The reference distribution used 24 degrees of freedom. Oversampled counts were not used in q. This correction inflated SE by approximately 2.6926 relative to the naive independent-fold calculation [10], [11]. It remains an approximation, especially for nonlinear rare-event AP, AUC and MCC [12].

One exploratory Holm family comprised 54 planned slots across AP, AUC, Brier and selected minority metrics plus limited interactions. Nine numerically equal or zero-variance slots did not receive artificial p-values; they were padded conservatively for adjustment. Marginal 95% intervals and separate Bonferroni family intervals were not mislabeled as Holm-adjusted intervals. A paired patient bootstrap retained all repeat predictions of each development patient together. Its fixed-score intervals omitted refitting and between-patient dependence from shared training and were not independent confirmatory evidence.

### H. Global and local explainability

Uncorrected LR, weighted LR and uncorrected RF were refitted on all development observations for exploratory explanation, not selected as winners. The same simple random sample of 256 real training observations, including eight events, was explained using two 32-row training backgrounds. Windows application control blocked SHAP/Numba binaries. The verified implementation therefore used exact analytical linear SHAP and exact enumeration of all 1,024 coalitions of the ten original-feature players for RF, rather than claiming a successful TreeExplainer run.

LR explanations were additive on decision-function log-odds: coefficient times transformed value minus background mean. Summing signed dummy contributions within an original variable gives the exact original-feature linear game. RF explained positive-class probability by averaging hybrid predictions over the finite background. Entire one-hot blocks were swapped jointly; all absent variables came from the same donor. This preserves valid categorical blocks but can break dependencies between present and absent clinical features. Global importance was mean absolute original-feature contribution; signed means were recorded separately. The finite-background game is not observational conditional or causal SHAP [14]. Additivity, grouping and independent coalition-formula replay were checked. Rank sensitivity used the alternate background and 1,000 conditional explanation-row resamples, not model refits.

## VI. Experimental Setup and Final-Evaluation Safeguards

The full imbalance run completed 625 fold fits and 511,000 OOF predictions. Source hashes, run manifests, warning/error records, sampling audits and OOF coverage checks isolated completed experiments from diagnostics. Continuous/categorical geometry and sampling targets were verified before the full run; preflight metrics were not promoted to publication results. Expected SVC deprecation warnings were allow-listed; unexpected warnings remained verification issues. Detailed environments and parameters are retained in the repository rather than inferred from library defaults at manuscript time.

The final comparison contained Dummy none, LR none, weighted LR, RF none and sigmoid-calibrated weighted LR. These jointly represented no-skill, linear/nonlinear probability references, weighting's detection trade-off and a calibration intervention. There was no composite clinical utility or sufficient evidence for a universal winner. All five streams retained cutoff 0.50; calibrated weighted LR also received a training-selected maximum-MCC cutoff of 0.061.

The protocol freeze was written on 10 November 2026 at 02:31:50 IST; fitted-model/calibrator hashes and the numerical cutoff were sealed at 02:31:57 IST. First locked API access followed at 02:33:15 IST. Both JSON records were exclusive-created, read-only and SHA-256 sealed, providing an application-level audit trail rather than adversarial immutable storage. Four final base pipelines were fitted to development data only. The final sigmoid reused verified repeat-one five-fold OOF weighted-LR decision scores, fitted one map to those scores/labels, and applied it to the full-development weighted base. No extra five-fold base refits were necessary. The additional cutoff maximized mean MCC across five separate saved nested-sigmoid OOF repeats, using the frozen grid/tie rule. This differs from calling its selected development maximum an unbiased performance estimate.

The locked test was loaded through the existing project API only after both seals and independent pre-access checks. Each base produced test probabilities once; sigmoid transformed the same weighted-base scores. The saved five streams generated the six predetermined operating rows. No post-test calibration, threshold adjustment, model reselection or additional comparison family was introduced. Calibration diagnostic regressions were descriptive score assessments, not deployed predictive fits.

Uncertainty used 5,000 paired ordinary test-patient resamples, with the same weights across streams, and exact Clopper-Pearson intervals for sensitivity, specificity and PPV conditional on their denominators. Intervals were marginal and conditional on fixed fits/selection, not simultaneous or full-training-process intervals. Undefined PPV draws were retained; a bootstrap PPV interval was unavailable if any draw was undefined. No patient bootstrap re-evaluated a model or accessed the spent raw test partition.

## VII. Results

### A. Baseline performance and minority detection

**TABLE II. Uncorrected development performance at 0.50; mean of five pooled-repeat metrics, not a mean of fold scores. Each repeat contains the same 4,088 observations and 199 events.**

| Model | AUC | AP | Recall | Spec. | MCC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy | 0.4979 | 0.0485 | 0.0000 | 1.0000 | 0.0000 | 0.0463 |
| LR | 0.8374 | 0.1689 | 0.0000 | 1.0000 | 0.0000 | 0.0430 |
| DT | 0.8018 | 0.1448 | 0.0040 | 0.9981 | 0.0055 | 0.0450 |
| RF | 0.8289 | 0.1617 | 0.0000 | 1.0000 | 0.0000 | 0.0428 |
| SVM | 0.6676 | 0.1022 | 0.0000 | 0.9996 | -0.0039 | 0.0459 |

Uncorrected LR and RF had useful ranking measures but zero pooled-repeat sensitivity at 0.50. DT detected a small fraction of events while remaining low in MCC. Dummy's within-fold ROC-AUC was 0.5; its pooled-repeat AUC 0.4979 reflects slightly different training priors across folds rather than a learned patient-level signal. This aggregation artifact is retained rather than silently reset to 0.5. A ranking score alone therefore does not establish useful event detection at the default cutoff.

### B. Effects of imbalance correction

**TABLE III. LR imbalance effects at 0.50; five pooled-repeat means, each based on 4,088 observations and 199 events. ROS/RUS denote random over-/undersampling; CatADASYN denotes custom CategoricalADASYN. PPV is recorded as zero when no positives are predicted in these CV summaries.**

| Strategy | AP | Recall | Spec. | PPV | Brier |
| --- | --- | --- | --- | --- | --- |
| None | 0.1689 | 0.0000 | 1.0000 | 0.0000 | 0.0430 |
| Weight | 0.1704 | 0.7930 | 0.7359 | 0.1332 | 0.1731 |
| ROS | 0.1691 | 0.6040 | 0.8347 | 0.1576 | 0.1125 |
| RUS | 0.1710 | 0.5879 | 0.8361 | 0.1551 | 0.1130 |
| SMOTENC | 0.1403 | 0.4854 | 0.8476 | 0.1401 | 0.1114 |
| CatADASYN | 0.1681 | 0.6111 | 0.8276 | 0.1535 | 0.1167 |

Table III quantifies the LR strategy effects on the same development partition.

LR weighting increased recall to 0.7930 while reducing specificity to 0.7359 and leaving low precision. RUS and ROS produced different detection/specificity compromises. SMOTENC and custom CategoricalADASYN did not uniformly dominate no correction. Within LR, the custom sampler's pooled-repeat AP was higher than SMOTENC's, but the associated corrected-CV AP/AUC comparisons did not survive Holm adjustment. These are model/strategy-specific effects, not evidence that more resampling always improves usefulness.

### C. Calibration behavior

**TABLE IV. Nested calibration; five pooled-repeat means, each based on 4,088 observations and 199 events. Weighted denotes class-weighted LR; RF denotes uncorrected Random Forest. Configuration selection was exploratory.**

| Configuration | AUC | AP | Brier | Log loss |
| --- | --- | --- | --- | --- |
| Weighted raw | 0.8364 | 0.1704 | 0.1731 | 0.5065 |
| Weighted sigmoid | 0.8364 | 0.1689 | 0.0428 | 0.1588 |
| Weighted isotonic | 0.8332 | 0.1616 | 0.0429 | 0.1672 |
| RF raw | 0.8289 | 0.1617 | 0.0428 | 0.1596 |
| RF sigmoid | 0.8263 | 0.1617 | 0.0441 | 0.1680 |

Table IV separates discrimination from probability error under nested calibration.

Weighted LR's nested pooled-repeat Brier decreased from 0.1731 to 0.0428 with sigmoid and 0.0429 with isotonic. Its sigmoid ROC-AUC remained close to the original value, while log loss improved. RF did not show a uniform calibration benefit; its sigmoid Brier was higher than its reference Brier. Pooled-repeat AUC/AP can change slightly under fold-specific monotone maps even when within-fold ranking is unchanged. These metrics must not be conflated with the single final map, which preserves the final base model's ranking.

Figure 1 shows held-out reliability and score distributions for the frozen representatives. Its bins are diagnostics rather than a new calibrator. Weighted raw probabilities are visibly on a different scale from observed frequencies; five and ten bins do not establish precise tail calibration with only 50 events.

### D. Threshold behavior and uncertainty

For the earlier inner-selected maximum-MCC sigmoid weighted-LR policy, pooled-repeat outer-validation means were sensitivity 0.6945, specificity 0.7792, precision 0.1389 and MCC 0.2376. These used fold-specific training-selected thresholds; their average is not a deployable cutoff, and they are not an unbiased estimate of the later exact full-development selector.

The 191-point development sweeps showed that increasing sensitivity generally incurs additional false positives and lower precision. Inner-selected policies were evaluated on outer rows not used to choose their thresholds; development constraints were not guaranteed in validation. The final 0.061 cutoff was chosen before test access and was not recalculated from test labels. Apparent development maxima are not substituted for protected policy estimates. Tables VI-VII distinguish probability performance from operating decisions; Figure 2 presents the actual final error counts and exact detection-rate intervals.

![Figure 1. Frozen-test five-quantile-bin reliability and score distributions for the five predefined streams (Step 39). Whiskers are marginal exact binomial event-fraction intervals; bins are diagnostic, not recalibration. Only 50 events limit tail interpretation.](figures/figure01.png)

![Figure 2. Frozen held-out operating points, exact sensitivity/specificity intervals and actual false-positive/false-negative counts (Step 39). The 0.061 policy was fixed using development data, not selected from this figure.](figures/figure02.png)

### E. Statistical comparisons and partition stability

**TABLE V. Exploratory paired fold effects. Differences are left minus right; higher Brier is worse. Intervals are marginal approximate corrected-CV 95% intervals from 25 paired, correlated folds; p-values share the 54-slot Holm family.**

| Contrast / metric | Difference | 95% interval | Holm p |
| --- | --- | --- | --- |
| LR weight - none: Brier | 0.1301 | 0.1225 to 0.1377 | 1.6e-20 |
| LR RUS - none: Brier | 0.0700 | 0.0614 to 0.0786 | 3.8e-13 |
| LR CatADASYN - SMOTENC: AUC | 0.0425 | 0.0168 to 0.0682 | 0.0797 |
| Weighted sigmoid - raw: Brier | -0.1302 | -0.1379 to -0.1226 | 1.98e-20 |
| Weight effect LR - RF: Brier | 0.0647 | 0.0601 to 0.0693 | 1.92e-18 |

Table V reports selected effects and their uncertainty without treating correlated folds as independent datasets.

Of 54 planned slots, 45 yielded informative approximate tests, nine were unavailable and 19 produced exploratory Holm-adjusted flags. No AP or AUC comparison survived adjustment. For custom CategoricalADASYN versus SMOTENC within LR, the AUC effect interval excluded zero marginally, but Holm p=0.0797; this is not multiplicity-adjusted significance. The weighting/Brier interaction indicates a larger probability-error penalty for LR than RF under the stated approximation. Failure to flag an effect is not equivalence.

Repeat-specific ranks also depended on strategy and partition. LR ranked first for AP in all five pooled repeats under none, weighting, ROS, SMOTENC and the custom sampler, but under RUS its ranks were 1, 1, 3, 1 and 2. These are descriptive ranks, not five independent replications. H4 was not tested by comparing fold SDs with pooled-repeat SDs, and seed 42 repetitions do not constitute an independently designed multi-seed study.

![Figure 3. Original-feature mean-absolute-SHAP shares normalized within each representative model (Step 38). LR log-odds and RF probability magnitudes are not cross-scale comparable. These are in-sample explanations, not causal effects.](figures/figure03.png)

### F. Global and local SHAP

Age ranked first for all three representatives. Its mean absolute contribution was 1.3451 log-odds in uncorrected LR, 1.5475 log-odds in weighted LR and 0.03403 probability units in RF. Their normalized shares were 68.75%, 63.89% and 42.81%, respectively. Figure 3 compares within-model relative shares, not cross-scale absolute magnitudes. BMI ranked tenth in uncorrected LR and fourth in RF; work type rose to second in weighted LR. This is descriptive compatibility with H6, not a formal hypothesis test or a clinical feature-importance ranking.

Primary-versus-alternate-background rank correlations were 0.9636, 0.9758 and 1.0000 for uncorrected LR, weighted LR and RF. The RF third/fourth positions were less stable to explanation-row composition: hypertension and BMI appeared in the top three in 56% and 44% of conditional resamples. Strong age dominance can coexist with uncertainty in neighboring lower ranks.

Local cases were the same three real training rows nearest RF sample probability quantiles 0.10, 0.50 and 0.90, without outcome filtering; all happened to be nonevents. They illustrate relative score strata, not positive-event exemplars or population-representative phenotypes. For row 3792, probabilities were 0.1435 for uncorrected LR, 0.8001 for weighted LR and 0.1415 for RF. Age was the largest positive contribution in each, but signed magnitudes had their respective output units. All explanations were in-sample and none described synthetic patients or causal effects.

### G. Frozen held-out evaluation

**TABLE VI. Frozen held-out probability performance; 1,022 observations, including 50 events and 972 nonevents. Threshold does not change these metrics.**

| Configuration | AUC | AP | Brier | Log loss |
| --- | --- | --- | --- | --- |
| Dummy | 0.5000 | 0.0489 | 0.0465 | 0.1953 |
| LR none | 0.8421 | 0.2700 | 0.0406 | 0.1541 |
| LR weighted | 0.8436 | 0.2682 | 0.1676 | 0.4929 |
| RF none | 0.8252 | 0.2133 | 0.0423 | 0.1611 |
| LR weighted + sigmoid | 0.8436 | 0.2682 | 0.0407 | 0.1539 |

**TABLE VII. Frozen held-out operating points; sensitivity denominator is 50 events and specificity denominator is 972 nonevents. PPV uses predicted positives, which vary by configuration. NA means no predicted positives; zero-denominator MCC uses the recorded zero convention.**

| Configuration | Cutoff | Recall | Spec. | PPV | F1 | MCC | Bal. acc. | FP/FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy | 0.500 | 0.0000 | 1.0000 | NA | 0.0000 | 0.0000 | 0.5000 | 0/50 |
| LR none | 0.500 | 0.0200 | 1.0000 | 1.0000 | 0.0392 | 0.1380 | 0.5100 | 0/49 |
| LR weighted | 0.500 | 0.8000 | 0.7428 | 0.1379 | 0.2353 | 0.2597 | 0.7714 | 250/10 |
| RF none | 0.500 | 0.0000 | 1.0000 | NA | 0.0000 | 0.0000 | 0.5000 | 0/50 |
| LR weighted + sigmoid | 0.500 | 0.0000 | 1.0000 | NA | 0.0000 | 0.0000 | 0.5000 | 0/50 |
| LR weighted + sigmoid / 0.061 | 0.061 | 0.8000 | 0.7778 | 0.1562 | 0.2614 | 0.2876 | 0.7889 | 216/10 |

Uncorrected LR's ROC-AUC 0.8421 and AP 0.2700 coexisted with sensitivity 0.0200 at 0.50: only one event was detected. Its precision of 1.0 rested on that single predicted positive and is not reliable evidence of high PPV. Dummy, RF and calibrated weighted LR at 0.50 predicted no positives; their PPV is undefined, not evidence of perfect precision.

Weighted LR at 0.50 detected 40 events, with 250 false positives and Brier 0.16763. Sigmoid-calibrated weighted LR had identical final ROC-AUC 0.8436 and AP 0.2682 but Brier 0.04073. At its frozen 0.061 cutoff, sensitivity was 0.8000 with exact 95% interval [0.6628, 0.8997]; specificity was 0.7778, precision 0.1562 and MCC 0.2876. This yielded 216 false positives and 10 false negatives. The precision exact interval was [0.1140, 0.2066].

For that calibrated stream, fixed-model bootstrap 95% intervals were [0.1693, 0.3882] for AP and [0.7845, 0.8972] for ROC-AUC. They do not incorporate refitting/selection uncertainty. Figure 4 contrasts development means/SD with these different test uncertainty summaries; an increase in held-out AP is not a statistically established improvement over CV or evidence for H4.

![Figure 4. Previously viewed development pooled-repeat mean and SD versus frozen-test estimates with fixed-model patient-bootstrap intervals (Step 39). Repeat SD is not a CI; differences are descriptive and do not test H4.](figures/figure04.png)

## VIII. Discussion

The main empirical result is the separation of discrimination, probability scale and decision behavior. An uncorrected model can rank observations reasonably well yet miss almost all recorded events at 0.50. Weighting can restore sensitivity while distorting natural-prevalence probabilities and increasing false positives. Calibration can repair much of that probability error without adding ranking information. The subsequent cutoff determines how that probability scale is converted into decisions.

The final weighted raw and sigmoid streams illustrate the distinction especially clearly. Their AUC and AP were identical, but their Brier scores differed substantially. At the same cutoff of 0.50, calibration changed the classifications from many positives to none. Comparing raw weighting at 0.50 with calibrated weighting at 0.061 additionally changes the threshold. The lower false-positive count at the latter operating point is therefore a joint property of the fitted score map and selected decision cutoff, not an isolated causal effect of calibration. In a single monotone-map model, changing probability cutoff corresponds to changing the raw-score operating point; calibration does not create a superior ROC curve.

The 216 false positives against 40 detected events at the lower frozen cutoff expose the workload implied by apparently attractive sensitivity. No clinical cost ratio, capacity constraint or prespecified minimal useful effect was available. Maximum development MCC is an exploratory statistical policy, not a validated clinical utility criterion. Its final results should be assessed alongside missed events, low PPV and uncertainty, not promoted as a recommended screening rule.

The statistical evidence also sets limits on ranking claims. Holm flags arose mainly for probability error and fixed-cutoff measures, whereas no AP/AUC comparison survived the specified adjustment. These results depend on an approximate repeated-CV variance model and do not establish universal superiority. Conditional bootstrap sensitivity and repeated sign consistency are not independent corroborating datasets. Previously viewed CV results also influenced the representative configuration set; nesting calibration does not eliminate that configuration-selection optimism.

For RQ6, common age dominance and differing lower ranks describe the learned functions under their chosen explanatory games. Correlation between age, work and marriage, a different link scale and distinct encodings can affect allocation. Attribution changes should not be interpreted as changed biological effects. The small background study improves transparency, but stable ranks across two backgrounds do not prove stable explanations under model refitting or another population.

The reproducibility contribution is concrete: common folds, fold-local transforms, explicit sampling geometry, preserved actual ratios, source/manifest hashes, available failure records and a test-access freeze. Those controls make results auditable. They do not validate the dataset's clinical provenance or erase the earlier full-dataset EDA. Computational isolation and evidence quality are complementary, not interchangeable.

The literature comparison supports a narrower interpretation than a claim of methodological precedence. The weighting/calibration separation complements probability-oriented GSM and Bayesian-network analyses [21], [22], while our paired strategy comparison extends the scope of these particular architecture-focused examples. Existing sampling and explainability studies [19], [23], [25], [28] already motivate examining model-dependent behavior. We do not compare their headline accuracies numerically with our test results: source subsets, prevalence after sampling, feature processing, tuning and validation are not interchangeable. Similar feature prominence across papers would be qualitative agreement, not independent replication or confirmation of a biological mechanism.

In a hypothetical screening workflow, missed positives could leave at-risk individuals unidentified, whereas excessive false positives could consume assessment capacity and expose individuals to unnecessary follow-up. These are reasons to report both error types, not demonstrated patient outcomes in this dataset. The retrospective stroke label cannot validate an acute-stroke triage service or a future-risk screening program. Calibration is relevant when scores are interpreted as probabilities, but clinical benefit additionally requires an appropriate outcome horizon, representative validation and explicit consequences of decisions [9], [32], [33].

## IX. Completed Empirical Outcomes and Hypothesis Evidence

The working hypotheses are retained with their original definitions, rather than reworded to fit the observations:

**H1:** Different ML models will show statistically meaningful differences across at least some discrimination, minority-class, calibration, or robustness metrics.

**H2:** Imbalance-handling strategy will materially change minority-class performance and may produce trade-offs between recall, precision and specificity.

**H3:** A model/configuration with strong ROC-AUC will not necessarily have the best probability calibration.

**H4:** Performance estimates from a single split will show greater apparent variability than estimates summarized across repeated stratified validation.

**H5:** Changing the classification threshold will produce meaningful sensitivity-specificity and false-positive/false-negative trade-offs.

**H6:** Important predictors identified by SHAP will be model-dependent rather than identical across every classifier.

**H7:** Observed model differences will vary with imbalance strategy, so model ranking based on one configuration will not necessarily persist across all configurations.

**H8:** Explicit dataset-quality and preprocessing checks will identify characteristics that materially affect interpretation or reproducibility of results.

This section replaces the proposal's Expected Outcomes. RQ1 receives partial internal evidence: model ranking, probability reliability and explainable fitted behavior were characterized, but broad robustness and clinical usefulness were not established. RQ2-RQ4 and RQ7 have descriptive benchmark, calibration and operating-point evidence. RQ5 is limited to observed partition behavior at one seed, RQ6 to the representative finite-background explanations, and RQ8 to qualified exploratory contrasts.

H1 has limited approximate comparative evidence, not a confirmatory global model-superiority result. H2 and H5 are descriptively compatible with observed recall/precision/specificity trade-offs, without a clinical materiality margin. H3 is compatible with high ranking performance and poor raw weighting reliability. H6 is descriptively compatible with model-dependent lower feature rankings; H7 has limited weighting/Brier interaction evidence plus strategy-specific rank changes. H4 remains untested because a direct, appropriate single-split-versus-repeated-estimation variability comparison was not performed.

H8 concerns whether explicit dataset-quality and preprocessing checks identify characteristics affecting interpretation or reproducibility. Missing BMI, Unknown smoking, severe imbalance, identifier exclusion and fold-local transforms provide relevant audit evidence, but the study did not randomize auditing or quantify its causal benefit. The accompanying matrix preserves the official wording of all eight hypotheses and eight questions, with separate evidence status and limits. Neither RQ1 nor H8 is labeled fully established beyond this dataset.

## X. Research Contribution

Relative to the overlapping integrated frameworks [21], [22], the completed contribution is a reproducible empirical evaluation in which discrimination, minority detection, probability accuracy, dependence-aware uncertainty, categorical synthesis, explanation and threshold decisions are reported together. It shows why high ranking performance, improved recall or a visually plausible explanation is insufficient by itself. It also supplies auditable implementations and clearly delimits the custom sampler from standard methods.

This is not an algorithmic novelty claim, a comprehensive superiority benchmark of all available learners or a new clinical risk score. The original proposal's boosting and independent-seed candidates were not retrospectively described as completed experiments. All preflight and exploratory artifacts remain distinct from the final frozen evaluation.

## XI. Scope and Limitations

First, this is a single publicly available dataset with limited clinical provenance, mixed ages, missing BMI and substantial Unknown smoking information. Median imputation and explicit category encoding preserve information mechanically but do not establish missing-at-random assumptions or population representativeness. No independent external cohort, validated risk horizon or prospective clinical evaluation was available.

Second, the held-out test contains only 50 positive observations. AP, sensitivity, PPV, reliability tails and diagnostic slopes are correspondingly uncertain. Exact binomial intervals are conditional on their denominators; bootstrap intervals assume exchangeable test patients and comparable source sampling. They omit training/configuration-selection uncertainty and are marginal rather than simultaneous. Degenerate bootstrap endpoints after no positive predictions do not prove zero uncertainty; exact detection intervals and undefined-PPV flags are retained.

Third, representative configurations were selected after development evidence was examined. Calibration was nested, but overall model/configuration selection was not. Historical full-dataset EDA preceded the partition lock and limits claims of complete test blindness. No post-test tuning occurred, and the spent partition was not accessed again during manuscript completion, but later safeguards cannot retroactively undo earlier aggregate exposure.

Fourth, repeated folds share training data and patients recur across repeats. Corrected-CV inference is approximate, with stronger assumptions for nonlinear rare-event metrics. Patient resampling of saved development predictions does not reproduce fitting-induced between-patient dependence. Independent seed sensitivity and H4 remain untested; neither a final split nor smaller repeat SDs establishes them.

Fifth, calibration and threshold analyses have limited event support, especially in tails. Flexible isotonic maps can be unstable with few events. Inner constraints may fail outside their selection data. The final threshold's selected development score is optimistic; its final test performance is a single fixed-policy estimate, not a clinical threshold validation.

Finally, SHAP used fitted-training explanations, a sample with eight events and two small backgrounds. Marginal group masking can generate clinically implausible cross-feature combinations despite valid categorical blocks. Different output scales and representations constrain comparison, and no model-refitting explanation stability or causal interpretation was established. No subgroup fairness or clinical net-benefit evaluation was performed.

The literature audit is bounded to the supplied set and selected methodological sources, with intermittent primary-site access limits. Source attribution does not guarantee matching cohorts, and several detailed validation procedures remain unresolved. Existing integrated frameworks preclude a first-of-kind claim.

## XII. Conclusion and Future Work

This completed study demonstrates why stroke-label prediction under severe imbalance requires evaluation beyond accuracy or ranking alone. Default-cutoff uncorrected LR detected only one held-out event despite AUC 0.8421. Weighting increased detection with many false positives and substantially poorer probability accuracy. Sigmoid calibration preserved weighted LR's ranking while improving Brier, and its frozen lower cutoff recovered sensitivity with a clear false-positive burden. Limited exploratory inference and representative exact SHAP complemented, rather than resolved, these trade-offs.

The methodological contribution, alongside existing integrated frameworks [21], [22], is a specific transparent common-protocol evaluation and reproducibility record, not a conclusively superior model or clinically validated system. Future work should prespecify meaningful-effect margins and a confirmatory family, evaluate independent model/CV seeds, directly design the H4 comparison, examine explanation stability under refitting, and validate externally in clinically representative temporal/geographic cohorts. Prospective assessment, outcome definition, subgroup performance and explicit decision utility are necessary before clinical use. Further development requires new independent evaluation data; the current test set is spent.

## Acknowledgment

The authors acknowledge the Department of Computer Science and Engineering (Artificial Intelligence), Indira Gandhi Delhi Technical University for Women, for the academic environment supporting this research.

## Data and Code Availability

The dataset is available from its public card [1]. Code and run-scoped artifacts are organized in the StrokeSure project (project URL: https://github.com/shreyarathore27/StrokeSure). The numerical provenance audit and supporting artifacts accompany this manuscript; public release of every local artifact is not implied by that URL. The claim audit links numerical statements and table cells to exact files and hashes. The final test was evaluated under the pre-test freezes; manuscript production used saved artifacts only and did not fit models or reload test observations.

## References

[1] fedesoriano, "Stroke Prediction Dataset," Kaggle, dataset. Accessed: Nov. 10, 2026. [Online]. Available: [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

[2] E. Dritsas and M. Trigka, "Stroke Risk Prediction with Machine Learning Techniques," Sensors, vol. 22, no. 13, art. 4670, 2022, [doi: 10.3390/s22134670](https://doi.org/10.3390/s22134670).

[3] T. Saito and M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," PLOS ONE, vol. 10, no. 3, e0118432, 2015, [doi: 10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).

[4] D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation," BMC Genomics, vol. 21, art. 6, 2020, [doi: 10.1186/s12864-019-6413-7](https://doi.org/10.1186/s12864-019-6413-7).

[5] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, 2002, [doi: 10.1613/jair.953](https://doi.org/10.1613/jair.953).

[6] H. He et al., "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," in 2008 IEEE International Joint Conference on Neural Networks, 2008, [doi: 10.1109/IJCNN.2008.4633969](https://doi.org/10.1109/IJCNN.2008.4633969).

[7] G. Lemaître, F. Nogueira, and C. K. Aridas, "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning," Journal of Machine Learning Research, vol. 18, no. 17, pp. 1-5, 2017. [Online]. Available: [https://www.jmlr.org/papers/v18/16-365.html](https://www.jmlr.org/papers/v18/16-365.html).

[8] A. Niculescu-Mizil and R. Caruana, "Predicting Good Probabilities With Supervised Learning," in Proceedings of the 22nd International Conference on Machine Learning, 2005. [Online]. Available: [https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf).

[9] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, "Calibration: the Achilles heel of predictive analytics," BMC Medicine, vol. 17, art. 230, 2019, [doi: 10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7).

[10] C. Nadeau and Y. Bengio, "Inference for the Generalization Error," Machine Learning, vol. 52, pp. 239-281, 2003, [doi: 10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366).

[11] R. R. Bouckaert and E. Frank, "Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms," 2004. Author manuscript, University of Waikato. [Online]. Available: [https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf).

[12] Y. Bengio and Y. Grandvalet, "No Unbiased Estimator of the Variance of K-Fold Cross-Validation," Journal of Machine Learning Research, vol. 5, pp. 1089-1105, 2004. [Online]. Available: [https://www.jmlr.org/papers/v5/grandvalet04a.html](https://www.jmlr.org/papers/v5/grandvalet04a.html).

[13] S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65-70, 1979. [Online]. Available: [https://www.jstor.org/stable/4615733](https://www.jstor.org/stable/4615733).

[14] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," 2017, arXiv:1705.07874. [Online]. Available: [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874).

[15] The imbalanced-learn developers, "SMOTENC," software documentation. Accessed: Nov. 10, 2026. [Online]. Available: [https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html).

[16] T. Tazin et al., "Stroke Disease Detection and Prediction Using Robust Learning Approaches," Journal of Healthcare Engineering, vol. 2021, art. 7633381, 2021, [doi: 10.1155/2021/7633381](https://doi.org/10.1155/2021/7633381).

[17] M. Al Duhayyim et al., "An Ensemble Machine Learning Technique for Stroke Prognosis," Computer Systems Science and Engineering, vol. 47, no. 1, pp. 413-429, 2023, [doi: 10.32604/csse.2023.037127](https://doi.org/10.32604/csse.2023.037127).

[18] S. Sahriar et al., "Unlocking stroke prediction: Harnessing projection-based statistical feature extraction with ML algorithms," Heliyon, vol. 10, no. 5, art. e27411, 2024, [doi: 10.1016/j.heliyon.2024.e27411](https://doi.org/10.1016/j.heliyon.2024.e27411).

[19] P. Chakraborty et al., "OptiSelect and EnShap: Integrating machine learning and game theory for ischemic stroke prediction," PLOS ONE, vol. 20, no. 8, art. e0328967, 2025, [doi: 10.1371/journal.pone.0328967](https://doi.org/10.1371/journal.pone.0328967).

[20] P. O. Akinwumi et al., "Evaluating machine learning models for stroke prediction based on clinical variables," Frontiers in Neurology, vol. 16, art. 1668420, 2025, [doi: 10.3389/fneur.2025.1668420](https://doi.org/10.3389/fneur.2025.1668420).

[21] R. Mustapha, A. Wusu, O. Olabanjo, and B. Adetunji, "GSM: An Integrated GAM-SHAP-MCDA Framework for Stroke Risk Assessment," Analytics, vol. 5, no. 1, art. 4, 2026, [doi: 10.3390/analytics5010004](https://doi.org/10.3390/analytics5010004).

[22] G. Pranil, R.-J. Singh, and R. Grewal, "Explainable AI for Conditional Stroke Probability Estimation: A Hybrid Bayesian Network Approach," Algorithms, vol. 19, no. 9, art. 736, 2026, [doi: 10.3390/a19090736](https://doi.org/10.3390/a19090736).

[23] M. Alruily et al., "A-Tuning Ensemble Machine Learning Technique for Cerebral Stroke Prediction," Applied Sciences, vol. 13, no. 8, art. 5047, 2023, [doi: 10.3390/app13085047](https://doi.org/10.3390/app13085047).

[24] I. G. Ivanov, Y. Kumchev, and V. J. Hooper, "An Optimization Precise Model of Stroke Data to Improve Stroke Prediction," Algorithms, vol. 16, no. 9, art. 417, 2023, [doi: 10.3390/a16090417](https://doi.org/10.3390/a16090417).

[25] Y. Dubey et al., "Explainable and Interpretable Model for the Early Detection of Brain Stroke Using Optimized Boosting Algorithms," Diagnostics, vol. 14, no. 22, art. 2514, 2024, [doi: 10.3390/diagnostics14222514](https://doi.org/10.3390/diagnostics14222514).

[26] D. Wu, X. Zhang, and X. Zhu, "A machine learning-based model for stroke prediction," Applied and Computational Engineering, vol. 67, pp. 122-130, 2024, [doi: 10.54254/2755-2721/67/20240645](https://doi.org/10.54254/2755-2721/67/20240645).

[27] K. Kitova, I. Ivanov, and V. Hooper, "Stroke Dataset Modeling: Comparative Study of Machine Learning Classification Methods," Algorithms, vol. 17, no. 12, art. 571, 2024, [doi: 10.3390/a17120571](https://doi.org/10.3390/a17120571).

[28] P. N. Srinivasu et al., "An Interpretable Approach with Explainable AI for Heart Stroke Prediction," Diagnostics, vol. 14, no. 2, art. 128, 2024, [doi: 10.3390/diagnostics14020128](https://doi.org/10.3390/diagnostics14020128).

[29] O. Shobayo, O. Zachariah, M. O. Odusami, and B. Ogunleye, "Prediction of Stroke Disease with Demographic and Behavioural Data Using Random Forest Algorithm," Analytics, vol. 2, no. 3, pp. 604-617, 2023, [doi: 10.3390/analytics2030034](https://doi.org/10.3390/analytics2030034).

[30] L. Mochurad et al., "Improving stroke risk prediction by integrating XGBoost, optimized principal component analysis, and explainable artificial intelligence," BMC Medical Informatics and Decision Making, vol. 25, art. 63, 2025, [doi: 10.1186/s12911-025-02894-z](https://doi.org/10.1186/s12911-025-02894-z). [RETRACTED Jun. 15, 2026; cited for integrity context only.]

[31] L. Mochurad et al., "Retraction Note: Improving stroke risk prediction by integrating XGBoost, optimized principal component analysis, and explainable artificial intelligence," BMC Medical Informatics and Decision Making, vol. 26, art. 217, Jun. 15, 2026, [doi: 10.1186/s12911-026-03630-x](https://doi.org/10.1186/s12911-026-03630-x).

[32] Centers for Disease Control and Prevention, "Risk Factors for Stroke," May 15, 2024. Accessed: Nov. 10, 2026. [Online]. Available: [https://www.cdc.gov/stroke/risk-factors/](https://www.cdc.gov/stroke/risk-factors/).

[33] World Health Organization, "Stroke," Dec. 19, 2025. Accessed: Nov. 10, 2026. [Online]. Available: [https://www.who.int/news-room/fact-sheets/detail/stroke](https://www.who.int/news-room/fact-sheets/detail/stroke).
