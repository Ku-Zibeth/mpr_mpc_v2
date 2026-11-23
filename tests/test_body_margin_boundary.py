from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from shapely.geometry import box
from env.road_boundary import BodyMarginRoadBoundary
from env.env import TDMPC2SafeMetaDriveEnv,SafeMetaDriveEnv
from tools.run_native_ablation100 import apply_environment_options,environment_protocol
from omegaconf import OmegaConf


def vehicle(y=0.,width=2.,heading=0.):
    lanes=[SimpleNamespace(shapely_polygon=box(0,-3.5,100,0)),SimpleNamespace(shapely_polygon=box(0,0,100,3.5))]
    # An unrelated opposite road must not make a departed body admissible.
    road_map=SimpleNamespace(road_network=SimpleNamespace(graph={'a':{'b':lanes},'opposite':{'end':[SimpleNamespace(shapely_polygon=box(0,3.5,100,20))]}}))
    f=np.array([np.cos(heading),np.sin(heading)]);n=np.array([-np.sin(heading),np.cos(heading)])
    points=[np.array([10,y])+x*2*f+z*width/2*n for x,z in [(1,1),(1,-1),(-1,-1),(-1,1)]]
    return SimpleNamespace(WIDTH=width,bounding_box=points,navigation=SimpleNamespace(map=road_map,checkpoints=['a','b']))


@pytest.mark.parametrize('y,expected',[(0,False),(2.5,False),(3.49,False),(3.5,False),(3.51,True),(-3.51,True)])
def test_touching_boundary_or_half_body_outside_is_allowed_but_more_is_rejected(y,expected):
    actual,margin=BodyMarginRoadBoundary().check(vehicle(y),.5)
    assert actual==expected and margin==1.


def test_heading_and_actual_width_are_used_not_center_only():
    assert BodyMarginRoadBoundary().check(vehicle(3.4,heading=np.pi/2),.5)[0]
    assert BodyMarginRoadBoundary().check(vehicle(3.5,width=4),.5)==(False,2.)
    assert BodyMarginRoadBoundary().check(vehicle(3.51,width=4),.5)[0]


def test_lane_dividers_do_not_count_as_road_edges_and_cache_expires_on_map_change():
    rule=BodyMarginRoadBoundary();v=vehicle();assert not rule.check(v,.5)[0]
    old=rule._road;assert not rule.check(v,.5)[0] and rule._road is old
    v2=vehicle(3.51);assert rule.check(v2,.5)[0] and rule._road is not old


def test_legacy_mode_still_delegates_to_native_predicate():
    fake=SimpleNamespace(config={'out_of_road_mode':'legacy'})
    with patch.object(SafeMetaDriveEnv,'_is_out_of_road',return_value=True):
        # Invoke the real subclass method with a real (uninitialized) instance.
        env=object.__new__(TDMPC2SafeMetaDriveEnv);env.config=fake.config
        assert env._is_out_of_road(object()) is True


def test_experiment_options_are_explicit_and_resume_protocol_detects_changes():
    raw=OmegaConf.create({'metadrive':{'simulator':{'traffic_density':.1,'accident_prob':.8,'out_of_road_mode':'legacy'}}})
    before=environment_protocol(raw);apply_environment_options(raw,.05,.5)
    after=environment_protocol(raw)
    assert before['traffic_density']==.1 and before['out_of_road_mode']=='legacy'
    assert after==dict(traffic_density=.05,accident_prob=.8,out_of_road_mode='body_margin',out_of_road_margin_widths=.5)
    with pytest.raises(ValueError):apply_environment_options(raw,1.1,.5)


def test_changed_departure_rule_prevents_old_results_import_even_with_same_initial_state(tmp_path):
    import json
    from tools.run_native_ablation100 import import_compatible
    source=tmp_path/'old';target=tmp_path/'new';source.mkdir();target.mkdir()
    (source/'launch_manifest.json').write_text('{}')
    old=OmegaConf.create({'metadrive':{'simulator':{'traffic_density':.05,'accident_prob':.8,'out_of_road_mode':'legacy'}}})
    new=OmegaConf.create(OmegaConf.to_container(old));apply_environment_options(new,.05,.5)
    OmegaConf.save(old,source/'launch_config.yaml');OmegaConf.save(new,target/'launch_config.yaml')
    with pytest.raises(ValueError,match='Environment/departure rules changed'):
        import_compatible(source,SimpleNamespace(out=target),{},[1],1)


@pytest.mark.parametrize('width',[1.852,4.])
def test_absolute_metre_margin_is_independent_of_vehicle_width(width):
    rule=BodyMarginRoadBoundary()
    within=vehicle(3.5-width/2+.09,width=width)
    outside=vehicle(3.5-width/2+.11,width=width)
    assert rule.check(within,margin_m=.1)==(False,.1)
    assert rule.check(outside,margin_m=.1)==(True,.1)
    with pytest.raises(ValueError,match='not both'):rule.check(within,.25,margin_m=.1)


def test_metre_options_are_recorded_and_switch_cleanly_to_width_mode():
    raw=OmegaConf.create({'metadrive':{'simulator':{'traffic_density':.05,'accident_prob':.8,'out_of_road_mode':'body_margin','out_of_road_margin_widths':.25}}})
    apply_environment_options(raw,.75,margin_m=.1)
    assert environment_protocol(raw)==dict(traffic_density=.75,accident_prob=.8,out_of_road_mode='body_margin',out_of_road_margin_widths=0.,out_of_road_margin_m=.1)
    absolute=environment_protocol(raw)
    apply_environment_options(raw,margin_widths=.25)
    assert environment_protocol(raw)!=absolute and raw.metadrive.simulator.out_of_road_margin_m is None
    with pytest.raises(ValueError,match='not both'):apply_environment_options(raw,margin_widths=.25,margin_m=.1)
    with pytest.raises(ValueError):apply_environment_options(raw,margin_m=float('nan'))
