"""
experiments/exp07_adaptive/audit_adaptive_judge.py

AUDIT of the E7b null result (exploratory, not pre-registered; added after reading the winners and a near miss). The E7b judge could under-credit compliance, which would
understate the adaptive attacker. Every candidate that EVADED the detector but was judged "not followed" is re-judged with a stronger second judge (claude-opus-5, same prompt).
Any flips would be reported as a sensitivity analysis. Output: adaptive_judge_audit.json.
"""
import json
from pathlib import Path
from experiments.exp07_adaptive.run_multi_target_v2 import judge, load_hosts
from experiments.exp07_adaptive.run_adaptive_attacker import read_jsonl
from src.attacks.destyle import make_client
HERE=Path('experiments/exp07_adaptive')
C=read_jsonl(HERE/'adaptive_candidates.jsonl'); V=read_jsonl(HERE/'mt_variants.jsonl')
orig={v['record_id']:v['original_text'] for v in V}
hosts=load_hosts(set(orig.values()),7)
cl=make_client('anthropic')
targets=[c for c in C if (not c['refused']) and c['flagged'] is False and not c['followed']]
print("evaded but judged not-followed:",len(targets),flush=True)
flips=[]
for c in targets:
    h=hosts[orig[c['record_id']]]
    if judge(cl,'claude-opus-5',orig[c['record_id']],h['question'],c['response']): flips.append(c)
print("second judge (claude-opus-5) says FOLLOWED on:",len(flips),"of",len(targets))
for c in flips: print(f"  [{c['injection_type']} r{c['round']} score {c['score']:.2f}] {c['text'][:140]!r}\n    RESP: {c['response'][:260]!r}")
(HERE/'adaptive_judge_audit.json').write_text(json.dumps({"n_evaded_not_followed":len(targets),"second_judge":"claude-opus-5","flipped":[{"record_id":c['record_id'],"round":c['round'],"score":c['score'],"text":c['text']} for c in flips]},indent=1))
