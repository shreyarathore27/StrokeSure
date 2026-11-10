"""Finish scientific interpretation and hash finalized artifacts after log closure."""
from pathlib import Path
import json,hashlib,csv
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=OUT/'scientific_shap_report.md';text=report.read_text(encoding='utf-8')
section='''## Numerical interpretation of RQ6/H6 and sensitivity

Age dominates all three representatives, but the importance allocation is model-specific. Primary-background mean absolute original-feature SHAP is 1.345068 log-odds for uncorrected LR (68.7452% of its total absolute importance), 1.547493 log-odds for weighted LR (63.8907%), and 0.034027 probability units for RF (42.8068%). These are separate output scales, not evidence that LR has a larger probability effect.

Uncorrected LR's next two features are average glucose (0.143797 log-odds; 7.3494%) and ever-married status (0.101859; 5.2059%). Weighted LR instead ranks work type second (0.193922; 8.0064%) and average glucose third (0.159732; 6.5948%). RF ranks average glucose second (0.010827 probability units; 13.6204%), hypertension third (0.006490; 8.1641%), and BMI fourth (0.006379; 8.0247%). BMI is tenth in uncorrected LR (0.005842 log-odds; 0.2986%) and ninth in weighted LR (0.040267; 1.6625%). Low fitted-model attribution is not evidence of clinical irrelevance.

Across ten feature ranks, Spearman correlations are 0.781818 (LR uncorrected/weighted), 0.442424 (LR uncorrected/RF), and 0.490909 (weighted LR/RF). Each pair shares two of its top three features. This is descriptive evidence compatible with H6: a shared dominant age predictor coexists with model-dependent lower rankings. It is not a significance test, and differences cannot be attributed solely to model family because output scales, fitted weighting and encodings also differ.

Changing the background retains age first for all three models. Primary-versus-alternate rank correlations are 0.963636, 0.975758 and 1.000000 respectively. Uncorrected LR's third-ranked feature switches from ever-married status to hypertension; weighted LR swaps work type and glucose at ranks two/three; RF's ten-feature order is unchanged. Maximum normalized importance-share shifts are 0.020737, 0.014164 and 0.015251. Two small backgrounds are a focused sensitivity check, not comprehensive reference-distribution robustness.

Across 1,000 conditional explanation-row resamples, age appears in the top three in 100% of draws for all models. Uncorrected LR's glucose top-three frequency is 100%, and ever-married status 90.5%; weighted LR's work type is 100% and glucose 97.3%. RF's glucose is 100%, while hypertension and BMI split the third position (56.0% and 44.0%). This close RF ranking should not be presented as a stable separation between hypertension and BMI. Percentiles and frequencies condition on the fitted model, chosen background and this explanation sample; they omit model-refitting uncertainty.

There are only eight events among 256 explained patients (3.125%), versus 199/4,088 (4.868%) in training. Backgrounds contain one and three events among 32 rows. Simple random sampling was deliberately not outcome-balanced: global means target overall sampled training-patient behavior and are dominated by nonevents. They are not stroke-case-specific importance estimates. The small references leave meaningful background uncertainty despite the observed rank agreement.

The three quantile-selected local rows (999, 2001, 3792) all have outcome zero. No outcome was used to select them, so they illustrate prediction strata rather than positive-event exemplars. At row3792, uncorrected LR, weighted LR and RF probabilities are 0.143475, 0.800092 and 0.141461. Age contributes positively by approximately 1.989 log-odds, 2.316 log-odds and 0.094 probability units after summing all features into the full output reconstruction (individual age bars and all ten signed values are in local_explanations.csv). The total contribution sums, not individual age contributions, are 1.9892, 2.3162 and 0.0943. This contrast illustrates model-specific score behavior, not calibrated clinical risk or a recommended cutoff.

All saved numeric CSVs use 17 significant digits. Read probability CSVs with pandas `float_precision="round_trip"` when replaying the nearest-quantile selection: default parsing can change the nearest observation at a near tie. Saved row IDs are authoritative; no local cases were replaced. The first delivery audit's two failures (active-log hash timing and default float parsing) are preserved in initial_delivery_verification.json and corrected in the final audit. No failed model fit or SHAP additivity check was removed.

'''
# Clarify that local values cited above are total contribution sums, not a single feature.
section=section.replace('Age contributes positively by approximately 1.989 log-odds, 2.316 log-odds and 0.094 probability units after summing all features into the full output reconstruction (individual age bars and all ten signed values are in local_explanations.csv). The total contribution sums, not individual age contributions, are 1.9892, 2.3162 and 0.0943.',
                        'Total signed contributions across all ten features are 1.9892 log-odds, 2.3162 log-odds and 0.0943 probability units, respectively; age is the largest positive contribution in each local explanation. Exact individual bars are in local_explanations.csv.')
if '## Numerical interpretation of RQ6/H6 and sensitivity' not in text:text=text.replace('## Model differences and sensitivity',section+'## Model differences and sensitivity',1)
text+='\n\nDelivery workflow: after the numerical analysis exits and its log closes, run `complete_report.py`, then `verify_delivery.py`, using the existing project Python. All eight PNG figures were visually reviewed.\n' if 'Delivery workflow:' not in text else ''
for a,b in {'seed42':'seed 42','max_iter1000':'max_iter=1000','RF200':'RF 200','min_samples_leaf5':'min_samples_leaf=5','Steps33':'Steps 33','Step33':'Step 33','of256':'of 256','contain32':'contain 32','of10':'of 10','contains1,000':'contains 1,000','same256':'same 256','seed38042':'seed 38042','sampling42':'sampling 42','alternate43':'alternate 43'}.items():text=text.replace(a,b)
report.write_text(text,encoding='utf-8')
mfile=OUT/'analysis_manifest.json';manifest=json.loads(mfile.read_text())
manifest['code_hashes']={p.name:sha(p) for p in OUT.glob('*.py')}
manifest['delivery_note']='Authored interpretation attached; finalized logs hashed after process exit. Initial delivery failures preserved, numerical outputs unchanged.'
mfile.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
rows=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and not any(x in p.parts for x in ['_dependencies','_matplotlib_config','__pycache__']) and p.name not in ['artifact_inventory.csv','delivery_verification.json']:
        rows.append(dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
with (OUT/'artifact_inventory.csv').open('w',newline='',encoding='utf-8') as h:
    writer=csv.DictWriter(h,fieldnames=['file','bytes','sha256']);writer.writeheader();writer.writerows(rows)
print('Finalized report and hashes:',len(rows),'artifacts')
