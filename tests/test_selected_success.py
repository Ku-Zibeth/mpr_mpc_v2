import csv,json
from pathlib import Path
from evaluation_success import apply_collision_failure

ROOT=Path(__file__).resolve().parents[1]

def test_success_is_arrival_and_whole_episode_collision_free_and_is_idempotent():
    for arrived,crashed,expected in [(1,0,1),(1,1,0),(0,1,0),(0,0,0)]:
        row=dict(final_success=arrived,episode_crash=crashed,return_value=123.,end_reason='arrive_dest',action=[.1,.2])
        apply_collision_failure(row);first=dict(row);apply_collision_failure(row)
        assert row==first and row['final_success']==expected and row['native_success']==arrived
        assert row['collision_and_native_success']==float(arrived and crashed)
        assert row['return_value']==123. and row['action']==[.1,.2]

def test_retained400_reference_rows_reproduce_the_selected_table():
    release=json.loads((ROOT/'models/manifest.json').read_text())
    rows=list(csv.DictReader((ROOT/'reports/selected_run/native_episodes.csv').open()))
    assert len(rows)==400
    for label,expected in release['strict_success_counts'].items():
        group=[r for r in rows if r['configuration']==label]
        assert len(group)==100
        assert sum(apply_collision_failure(r)['final_success'] for r in group)==expected
