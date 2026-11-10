"""Attach authored scientific interpretation and refresh this analysis's hashes."""
from pathlib import Path
import csv,json,hashlib

out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=out/'scientific_statistical_robustness_report.md'
text=report.read_text(encoding='utf-8')
interpretation='''## Key numerical findings and interpretation

These are mean paired fold effects, with marginal approximate corrected-CV 95% intervals unless stated otherwise. Adjusted p-values use the entire 54-slot exploratory family. They remain conditional on the variance approximation; none constitutes preregistered confirmation.

* **LR weighting versus none:** AP difference −0.001126 [−0.017579, 0.015328] and ROC-AUC −0.000976 [−0.003954, 0.002003], both Holm p=1.000. Sensitivity increased 0.792897 and specificity decreased 0.264130 at 0.50; Brier worsened by 0.130083 [0.122514, 0.137652], Holm p=1.596e−20. The substantial minority-detection gain therefore comes with additional false positives and poorer probability accuracy, without demonstrated discrimination improvement.
* **LR random undersampling versus none:** AP −0.001402 [−0.026331, 0.023526], ROC-AUC −0.004221 [−0.011038, 0.002597], both Holm p=1.000. Sensitivity increased 0.587769, specificity decreased 0.163899 and Brier worsened by 0.070031 [0.061434, 0.078628]. This is another trade-off rather than unqualified improvement.
* **RF versus LR with no correction:** AP −0.011284 [−0.043898, 0.021330], ROC-AUC −0.008127 [−0.025065, 0.008812], Brier −0.000134 [−0.001022, 0.000753]; all Holm p=1.000. No difference is established by these comparisons, and this does not establish equivalence. Equal fixed-threshold sensitivity/specificity/MCC differences have no informative inferential variance.
* **Custom categorical-aware ADASYN versus SMOTENC within LR:** AP +0.027391 [−0.005179, 0.059962], Holm p=1.000; ROC-AUC +0.042505 [0.016811, 0.068199], Holm p=0.079671; Brier +0.005322 [0.002091, 0.008553], Holm p=0.080173. Sensitivity increased 0.125410, while specificity decreased 0.020005 (Holm p=0.022595). The apparent discrimination gains do not survive the planned multiplicity adjustment. This comparison concerns the project's custom sampler, not stock ADASYN.
* **Nested calibration of weighted LR:** sigmoid Brier changed −0.130230 [−0.137883, −0.122576], Holm p=1.978e−20; isotonic −0.130163 [−0.137728, −0.122599], Holm p=1.579e−20. Sigmoid preserves within-fold AP/AUC to numerical tolerance; isotonic AP changes −0.018710, Holm p=0.858761. At the unchanged 0.50 cutoff, sigmoid sensitivity decreases 0.792897 and isotonic 0.791897. This reflects probability-scale and decision-cutoff changes and is not evidence that better probability accuracy necessarily improves detection at 0.50, or that calibration itself is clinically harmful. Brier is not a pure calibration metric.
* **Nested calibration of uncorrected RF:** sigmoid Brier increases 0.001218 [0.000552, 0.001883], Holm p=0.033254. Isotonic Brier changes only +0.000038 [−0.000248, 0.000324], Holm p=1.000. Calibration is therefore not uniformly beneficial across model/configuration choices.
* **Limited model-by-strategy evidence:** the weighting Brier interaction, (weighting−none in LR) minus (weighting−none in RF), is +0.064696 [0.060060, 0.069332], Holm p=1.915e−18. None of the tested AP/AUC interaction slots passes Holm adjustment. This indicates a larger weighting-related Brier penalty in LR under the approximation; it is not a global interaction claim across every model and strategy.

The patient-bootstrap tables use a different estimand, the mean of five pooled-repeat metrics. For example, LR weighting has pooled-repeat AP effect +0.001565, versus fold-mean −0.001126. Neither aggregation is silently substituted for the other. Custom ADASYN's pooled-repeat AP/AUC gains are positive in all five repeats, while LR weighting and undersampling AP gains each reverse direction in one repeat. These are descriptive partition-consistency statements, not independent replication counts.

For AP ranks among four learned families, LR ranks first in all five pooled repeats under none, weighting, random oversampling, SMOTENC and custom ADASYN. Under undersampling its ranks are 1, 1, 3, 1, 2; SVC leads repeat 3 and RF repeat 5. Consequently, the apparent ranking depends on both strategy and partition. These observations address RQ5 and the descriptive component of H7 without establishing independent seed robustness.

RQ8 and H1/H2 receive limited exploratory evidence for probability accuracy and fixed-cutoff trade-offs, not a universal model winner or clinically meaningful superiority. H7 receives the limited weighting/Brier interaction evidence above plus descriptive rankings. H4 remains untested. Independent seed sensitivity and a defensible single-split-versus-repeated-estimate comparison require a separately designed future experiment.

'''
if '## Key numerical findings and interpretation' not in text:
    text=text.replace('## Exact paired observation structure',interpretation+'## Exact paired observation structure',1)
text=text.replace('Reproduce with the existing project Python:',
                  'Reproduce numerical outputs with the existing project Python, then run `results/step37_statistical_robustness/complete_report.py` to attach the authored interpretation and refresh hashes:')
report.write_text(text,encoding='utf-8')
manifest_path=out/'analysis_manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['code_hashes']={p.name:sha(p) for p in out.glob('*.py')}
manifest['report_completion']='Authored numerical interpretation attached by complete_report.py; numerical outputs unchanged.'
manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
inventory=[]
for p in sorted(out.rglob('*')):
    if p.is_file() and not any(x in p.parts for x in ['_matplotlib_config','__pycache__']) and p.name not in ['artifact_inventory.csv','execution.log','delivery_verification.json']:
        inventory.append(dict(file=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
with (out/'artifact_inventory.csv').open('w',newline='',encoding='utf-8') as handle:
    writer=csv.DictWriter(handle,fieldnames=['file','bytes','sha256']);writer.writeheader();writer.writerows(inventory)
print('Report interpretation complete; inventory refreshed:',len(inventory),'artifacts')
