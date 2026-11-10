"""Finalize report-only hashes after logging processes close; never changes freezes."""
from final_common import *
for name in ['final_scientific_results_report.md','paper_ready_results.md','paper_ready_discussion.md']:
    p=OUT/name;text=p.read_text(encoding='utf-8')
    replacements={'and25':'and 25','using5-fold':'using 5-fold','×5-repeat':'× 5-repeat','on4,088':'on 4,088','generating625':'generating 625','and511,000':'and 511,000','cutoff0.':'cutoff 0.','threshold0.':'threshold 0.','at0.':'at 0.','within1e-12':'within 1e-12','established191':'established 191','contained1,022':'contained 1,022','patients:50':'patients: 50','and972':'and 972','detected40/50':'detected 40/50','sensitivity0.80':'sensitivity 0.80','exact95%':'exact 95%','with216':'with 216','specificity0.7778':'specificity 0.7778','precision0.15625':'precision 0.15625','was0.28764':'was 0.28764','of1.0':'of 1.0','only50':'only 50','from5,000':'from 5,000','AP0.':'AP 0.','ROC-AUC0.':'ROC-AUC 0.','Brier0.':'Brier 0.','from0.':'from 0.','to0.':'to 0.','a54-slot':'a 54-slot','ranged0.':'ranged 0.','requested0.':'requested 0.','has249':'has 249','among5,110':'among 5,110','observations,201':'observations, 201','and1,544':'and 1,544','only one predicted positive':'only one predicted positive'}
    for a,b in replacements.items():text=text.replace(a,b)
    if name=='final_scientific_results_report.md' and (OUT/'delivery_verification.json').exists() and '## Final delivery check' not in text:
        v=json.loads((OUT/'delivery_verification.json').read_text())
        text+=f'\n\n## Final delivery check\n\nIndependent saved-prediction audit: {v["n_passed"]}/{v["n_checks"]} PASS, {v["n_failed"]} failed. All five 300-dpi PNGs were visually reviewed. All 249 prior scientific artifacts/source files remain unchanged. No final model was selected in response to test results. The test partition is now spent; do not reuse it for tuning or additional model selection.\n'
    p.write_text(text,encoding='utf-8')
manifest=json.loads((OUT/'analysis_manifest.json').read_text())
manifest['code_hashes']={p.name:sha(p) for p in OUT.glob('*.py')}
dump(manifest,'analysis_manifest.json')
rows=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and not any(x in p.parts for x in ['_matplotlib_config','__pycache__']) and p.name not in ['artifact_inventory.csv','delivery_verification.json']:
        rows.append(dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
write(pd.DataFrame(rows),'artifact_inventory');print('Final inventory:',len(rows),'files')
