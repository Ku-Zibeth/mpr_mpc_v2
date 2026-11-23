"""Rebuild the retained four-way report with collision-as-failure statistics."""
from pathlib import Path
import argparse,csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation_success import apply_collision_failure
from evaluation_statistics import summarize_runs,paired_comparisons


def csv_rows(path):
    with Path(path).open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            if value=='':row[key]=None
            else:
                try:row[key]=float(value)
                except (TypeError,ValueError):pass
        for key in ['episode','scenario','test_seed','episode_seed','traffic_seed','length']:
            if row.get(key) is not None:row[key]=int(row[key])
        apply_collision_failure(row)
    return rows


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'reports/selected_run/native_episodes.csv')
    ap.add_argument('--output',type=Path,default=ROOT/'reports/selected_run')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=csv_rows(args.input);release=json.loads((ROOT/'models/manifest.json').read_text())
    groups=list(release['strict_success_counts']);summaries=[];failures={};matrix=[]
    if len(rows)!=400 or len({(r['configuration'],r['scenario']) for r in rows})!=400:raise ValueError('Reference requires400 unique rows')
    for label in groups:
        group=[r for r in rows if r['configuration']==label]
        summary=summarize_runs(label,group,samples=2000,seed=20261124)
        if summary['episodes']!=100 or round(summary['success']*100)!=release['strict_success_counts'][label]:raise ValueError(f'Reference mismatch: {label}')
        summaries.append(summary);failed=[r for r in group if r['final_success']<.5];failures[label]=[r['scenario'] for r in failed]
        directory=out/'failures';directory.mkdir(exist_ok=True);write_csv(directory/f'{label}.csv',failed)
        (directory/f'{label}_seeds.json').write_text(json.dumps(failures[label],indent=2))
    write_csv(out/'episodes.csv',rows);write_csv(out/'summary.csv',summaries)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    (out/'failures/all_failed_seeds.json').write_text(json.dumps(failures,indent=2))
    write_csv(out/'paired_comparisons.csv',paired_comparisons(rows,samples=2000,seed=20261124))
    text=['# Retained model: collision-free completion','',
          'Success = native destination reached AND no collision at any point in the episode. An episode that both collides and fails is counted once.',
          'Native termination, complete-episode return, safety cost and controls are unchanged. Native arrival is retained as `native_success`; `final_success` is the strict endpoint.',
          'Environment: traffic_density=0.075; accident_prob=0.8; body departure tolerance=0.1m. Same100 scenarios and seed20261127 for all groups.','',
          '| Group | Native arrival | Collision then arrival | Strict success | Failure |',
          '|---|---:|---:|---:|---:|']
    for r in summaries:text.append(f"| {r['configuration']} | {r['native_success_count']}% | {r['collision_and_native_success_count']} | {100*r['success']:.0f}% | {100-round(r['success']*100)} |")
    text+=['','This is the retained result on previously used maps, not a new independent validation. The source checkpoint and native400-episode CSV are retained for reproducibility.','',
           'Rebuild this report: `python tools/summarize_selected.py`. Run a fresh evaluation: `bash scripts/evaluate_selected.sh`.']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print({r['configuration']:round(r['success']*100) for r in summaries},flush=True)

if __name__=='__main__':main()
