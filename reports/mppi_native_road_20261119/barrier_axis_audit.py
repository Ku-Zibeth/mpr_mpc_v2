"""Code-level footprint counterexample, not a new driving experiment or controller."""
from study import *
from shapely.geometry import box
if __name__=='__main__':
    source=Path('/home/kzb/anaconda3/envs/tdmpc2/lib/python3.11/site-packages/metadrive/component/static_object/traffic_object.py')
    # Installed TrafficBarrier body is WIDTH on localX and LENGTH on localY;
    # BaseStaticObject uses BaseObject heading, while vehicle overrides +pi/2.
    # Planner assumes LENGTH is aligned with obj.heading_theta for every object.
    from metadrive.component.static_object.traffic_object import TrafficBarrier
    from metadrive.base_class.base_object import BaseObject
    assert TrafficBarrier.heading_theta is BaseObject.heading_theta
    assert 'BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.height / 2))' in source.read_text()
    length=float(TrafficBarrier.LENGTH.fget(None));width=float(TrafficBarrier.WIDTH.fget(None))
    ego_length,ego_width=4.515,1.852;ego_y=1.6;margin=.3
    actual=box(-width/2,-length/2,width/2,length/2)
    modeled=box(-length/2,-width/2,length/2,width/2)
    ego=box(-ego_length/2,ego_y-ego_width/2,ego_length/2,ego_y+ego_width/2)
    inflated_ego=box(-ego_length/2-margin,ego_y-ego_width/2-margin,ego_length/2+margin,ego_y+ego_width/2+margin)
    assert actual.intersects(ego) and not modeled.intersects(inflated_ego)
    dump(R/'barrier_axis_audit.json',dict(source=str(source),source_sha256=file_sha(source),planner_source_sha256=file_sha(ROOT/'lattice/frenet_metadrive.py'),
        dimensions=dict(length=length,width=width),body_local_extents='BulletBoxShape(WIDTH/2,LENGTH/2,height/2)',reported_heading='BaseObject localX yaw, no vehicle-style +pi/2 override',
        planner_assumption='LENGTH aligned with reported heading',axis_disagreement_degrees=90,synthetic_true_overlap=True,synthetic_legacy_overlap_even_with_model_margin=False,
        implication='Static barrier geometry is not guaranteed consistent with the existing generic obstacle heading convention. This is separate from WM prediction error.',
        existing_policy_input_unchanged=True,not_fixed_inside_validated_variant=True,independent_result_does_not_cover_a_future_fix=True,
        introduced_barrier_crashes=[[500129,1],[500322,1]],causal_limit='Native replays identify barrier contacts. This code-level counterexample proves a geometry defect, but does not prove fixing it alone prevents those episodes.'))
