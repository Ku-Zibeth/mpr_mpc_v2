"""MetaDrive原生风险场计算模块

功能说明：
    该模块从MetaDrive实时读取车道、路网、车辆和交通障碍物状态，计算标量风险成本。
    不依赖任何渲染器或硬编码的高速公路几何信息，完全基于MetaDrive原生API。

核心特性：
    - 动态获取真实道路几何（车道宽度、边界距离）
    - 支持Frenet坐标系转换（适配弯道）
    - 区分车辆和静态障碍物的风险建模
    - 引入车头时距(Headway)和碰撞时间(TTC)预测性安全指标
    - 可配置化参数管理，支持热调整
"""

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

# 尝试导入MetaDrive组件（支持多种安装方式）
try:
    from metadrive.constants import PGLineColor, PGLineType
    from metadrive.component.static_object.traffic_object import TrafficObject
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
except ImportError:  # 本地可编辑树也可能以metadrive.metadrive形式导入
    try:
        from metadrive.metadrive.constants import PGLineColor, PGLineType
        from metadrive.metadrive.component.static_object.traffic_object import TrafficObject
        from metadrive.metadrive.component.vehicle.base_vehicle import BaseVehicle
    except ImportError:
        PGLineColor = None
        PGLineType = None
        TrafficObject = None
        BaseVehicle = None


class RiskFieldCalculator:
    """从MetaDrive原生场景信息计算标量风险成本
    
    该类实现了完整的安全风险场模型，包含以下组件：
    1. 道路边界风险：防止偏离道路
    2. 车道线风险：鼓励保持在车道中心
    3. 偏离道路风险：惩罚驶出可行驶区域
    4. 周围车辆风险：考虑车辆尺寸和相对速度
    5. 静态障碍物风险：避让交通锥、路障等
    6. 车头时距风险：跟车过近的惩罚
    7. 碰撞时间风险：预测性碰撞预警
    """

    # 默认配置参数
    DEFAULTS = {
        "risk_field_max_distance": 50.0,              # 最大感知距离（米）
        "risk_field_boundary_weight": 2.0,            # 道路边界权重
        "risk_field_lane_weight": 0.1,                # 车道线权重
        "risk_field_offroad_weight": 2.0,             # 偏离道路权重
        "risk_field_vehicle_weight": 2.0,             # 周围车辆权重
        "risk_field_object_weight": 1.0,              # 静态障碍物权重
        "risk_field_headway_weight": 0.0,             # 车头时距权重 0
        "risk_field_ttc_weight": 0.0,                 # 碰撞时间权重 0
        "risk_field_boundary_sigma": 0.75,            # 边界风险扩散系数（米）
        "risk_field_lane_edge_sigma": 0.75,           # 车道线风险基础sigma（用于窄核+宽肩双层分布）
        "risk_field_broken_line_sigma": 0.1,        # 虚线风险扩散系数（米），单独控制宽度
        "risk_field_lane_core_sigma_scale": 2.2,    # 车道线强惩罚窄核sigma缩放（只在线心附近显著增大）
        "risk_field_lane_shoulder_sigma_scale": 1.5, # 车道线弱惩罚宽肩sigma缩放（扩大感知范围但保持温和）
        "risk_field_lane_shoulder_weight": 0.9,     # 车道线宽肩占比（其余权重分配给窄核）
        "risk_field_broken_line_factor": 0.1,        # 虚线换道线风险系数（允许跨越，低风险）  5-
        "risk_field_solid_line_factor": 1.0,         # 普通实线风险系数（不鼓励跨越）
        "risk_field_boundary_line_factor": 1.0,       # 道路边界/护栏线风险系数
        "risk_field_oncoming_line_factor": 1.50,      # 黄色/对向分隔线风险系数（最高）
        "risk_field_offroad_cost": 1.0,               # 路外边缘带基础成本
        "risk_field_offroad_sigma": 1.0,              # 路外边缘带宽度（米）
        "risk_field_on_lane_margin": 0.05,            # 在车道上的容差（米）
        "risk_field_vehicle_longitudinal_sigma": 6.8, # 车辆纵向风险扩散（米）
        "risk_field_vehicle_lateral_sigma": 2.0,      # 车辆横向风险扩散（米）
        "risk_field_vehicle_beta": 2,               # 车辆超高斯指数（控制形状）
        "risk_field_vehicle_dynamic_sigma_scale": 1.2,# 动态风险sigma缩放
        "risk_field_vehicle_dynamic_alpha": 0.35,      # 动态风险不对称系数
        "risk_field_vehicle_min_dynamic_sigma": 0.5,  # 最小动态sigma
        "risk_field_object_longitudinal_sigma": 5.5,  # 障碍物纵向风险扩散（米）
        "risk_field_object_lateral_sigma": 2.0,       # 障碍物横向风险扩散（米）
        "risk_field_object_beta": 2.0,                # 障碍物超高斯指数
        "risk_field_lane_beta": 2.0,                  # 车道线超高斯指数（控制边缘惩罚陡峭程度）
        "risk_field_headway_time_threshold": 1.2,     # 车头时距阈值（秒）
        "risk_field_ttc_threshold": 3.0,              # 碰撞时间阈值（秒）
        "risk_field_min_speed": 0.5,                  # 最小速度（避免除零）
        "risk_field_headway_cost_clip": 1.0,          # 车头时距成本上限
        "risk_field_ttc_cost_clip": 1.0,              # TTC成本上限
        "risk_field_raw_clip": 10.0,                  # 原始风险成本上限
    }

    EPS = 1e-6  # 数值稳定性常数

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化风险场计算器
        
        Args:
            config: 配置字典，会覆盖默认参数
        """
        self.config = config or {}

    def calculate(self, env: Any, vehicle: Any) -> Tuple[float, Dict[str, Any]]:
        """计算给定主车的风险成本
        
        Args:
            env: MetaDrive环境实例
            vehicle: 主车对象
            
        Returns:
            (risk_cost, info) 元组，其中：
            - risk_cost: 总风险成本（标量）
            - info: 详细的风险分解信息字典
        """
        # 获取当前车道
        lane = self._current_lane(env, vehicle)
        
        # 分别计算各类风险
        road_cost, road_info = self._road_risk(env, vehicle, lane)
        _, vehicle_info = self._vehicle_risk(env, vehicle, lane)
        _, object_info = self._object_risk(env, vehicle, lane)

        # 提取各组件成本
        boundary_cost = road_info["risk_field_boundary_cost"]
        lane_cost = road_info["risk_field_lane_cost"]
        offroad_cost = road_info["risk_field_offroad_cost"]
        headway_cost = vehicle_info["risk_field_headway_cost"]
        ttc_cost = vehicle_info["risk_field_ttc_cost"]
        surrounding_vehicle_cost = vehicle_info["risk_field_vehicle_cost"]
        surrounding_object_cost = object_info["risk_field_object_cost"]

        # ========== 加权求和得到总风险 ==========
        risk_cost = (
            self._cfg("risk_field_boundary_weight") * boundary_cost
            + self._cfg("risk_field_lane_weight") * lane_cost
            + self._cfg("risk_field_offroad_weight") * offroad_cost
            + self._cfg("risk_field_vehicle_weight") * surrounding_vehicle_cost
            + self._cfg("risk_field_object_weight") * surrounding_object_cost
            + self._cfg("risk_field_headway_weight") * headway_cost
            + self._cfg("risk_field_ttc_weight") * ttc_cost
        )
        
        # 限制为非负数并裁剪到上限
        risk_cost = self._clip_nonnegative(risk_cost, self._cfg("risk_field_raw_clip"))

        # 构建返回信息
        info = {
            "risk_field_cost": float(risk_cost),
            "risk_field_road_cost": float(road_cost),
        }
        info.update(road_info)
        info.update(vehicle_info)
        info.update(object_info)
        return float(risk_cost), info

    def calculate_at_position(
        self,
        env: Any,
        source_vehicle: Any,
        position: Any,
        *,
        speed: Optional[float] = None,
        use_closest_lane: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """在任意BEV/世界坐标位置评估相同的风险模型
        
        该函数用于调试可视化。采样车辆保持主车的尺寸和速度，
        同时将查询点在路网上移动。
        
        Args:
            env: MetaDrive环境
            source_vehicle: 源车辆（提供尺寸和速度参考）
            position: 查询位置 [x, y]
            speed: 可选的速度覆盖值
            use_closest_lane: 是否使用最近车道而非导航车道
            
        Returns:
            (risk_cost, info) 元组
        """
        # 创建代理车辆对象
        sample_vehicle = _RiskFieldSampleVehicle(
            source_vehicle,
            position,
            speed=speed,
            use_closest_lane=use_closest_lane,
        )
        return self.calculate(env, sample_vehicle)

    def _road_risk(self, env: Any, vehicle: Any, lane: Any) -> Tuple[float, Dict[str, Any]]:
        """计算道路相关风险（边界+车道+偏离道路）
        
        Args:
            env: MetaDrive环境
            vehicle: 主车对象
            lane: 当前车道对象
            
        Returns:
            (road_cost, info) 元组
        """
        ego_pos = self._xy(getattr(vehicle, "position", (0.0, 0.0)))
        lane_index = self._lane_index(lane)
        lane_longitudinal = math.nan
        lateral_offset = math.nan
        lane_width = math.nan
        lane_cost = 0.0
        lane_core_cost = 0.0
        lane_shoulder_cost = 0.0
        offroad_cost = 0.0
        side0_line_profile = {"kind": "none", "factor": 0.0, "sigma": self._cfg("risk_field_lane_edge_sigma")}
        side1_line_profile = {"kind": "none", "factor": 0.0, "sigma": self._cfg("risk_field_lane_edge_sigma")}

        # ========== 计算车道偏离风险 ==========
        if lane is not None:
            # 转换到Frenet坐标系
            lane_longitudinal, lateral_offset = self._lane_coordinates(lane, ego_pos)
            lane_width = self._lane_width(lane, lane_longitudinal)
            if math.isfinite(lateral_offset) and math.isfinite(lane_width) and lane_width > self.EPS:
                # MetaDrive的side=0/1分别对应车道两侧边线。不同线型使用不同风险系数：
                # 虚线低、普通实线高、边界高、黄色/对向分隔线最高。
                side0_line_profile = self.lane_line_risk_profile(lane, 0)
                side1_line_profile = self.lane_line_risk_profile(lane, 1)
                side_0_gap = max(lateral_offset + lane_width / 2.0, 0.0)
                side_1_gap = max(lane_width / 2.0 - lateral_offset, 0.0)
                # 采用“窄核 + 宽肩”的双层分布：
                # 宽肩提供更远的低强度预警，窄核保证只有贴近车道线中心时才出现大惩罚。
                side0_components = self.lane_line_risk_components(side_0_gap, side0_line_profile)
                side1_components = self.lane_line_risk_components(side_1_gap, side1_line_profile)
                lane_cost = max(side0_components["total"], side1_components["total"])
                lane_core_cost = max(side0_components["core"], side1_components["core"])
                lane_shoulder_cost = max(side0_components["shoulder"], side1_components["shoulder"])

        # ========== 计算偏离道路风险 ==========
        on_road, offroad_distance = self._lane_surface_state(env, ego_pos)
        if not on_road:
            # 只在真实道路边缘附近形成风险带，避免把整片路外区域全部涂满。
            offroad_cost = self._cfg("risk_field_offroad_cost") * self._one_dimensional_risk(
                offroad_distance,
                self._cfg("risk_field_offroad_sigma"),
            )

        # ========== 计算道路边界风险 ==========
        left_dist, right_dist = self._route_boundary_distances(vehicle)
        if not (math.isfinite(left_dist) and math.isfinite(right_dist)):
            # 回退方案：用车道宽度估算
            left_dist, right_dist = self._lane_boundary_fallback(lane, lane_longitudinal, lateral_offset)

        boundary_cost = 0.0
        if math.isfinite(left_dist) and math.isfinite(right_dist):
            min_boundary_distance = min(left_dist, right_dist)
            if min_boundary_distance < 0.0:
                # 已经越过边界时也只画边缘带，避免离道路越远越红、整片铺满。
                boundary_cost = self._one_dimensional_risk(
                    abs(min_boundary_distance),
                    self._cfg("risk_field_boundary_sigma"),
                )
            else:
                # 高斯风险：越靠近边界风险越高
                boundary_cost = self._one_dimensional_risk(
                    min_boundary_distance,
                    self._cfg("risk_field_boundary_sigma"),
                )

        # 边界成本至少为偏离道路成本
        boundary_cost = max(boundary_cost, offroad_cost)
        road_cost = boundary_cost + lane_cost + offroad_cost
        
        return float(road_cost), {
            "risk_field_boundary_cost": float(boundary_cost),
            "risk_field_lane_cost": float(lane_cost),
            "risk_field_lane_core_cost": float(lane_core_cost),
            "risk_field_lane_shoulder_cost": float(lane_shoulder_cost),
            "risk_field_offroad_cost": float(offroad_cost),
            "risk_field_on_road": bool(on_road),
            "risk_field_offroad_distance": self._finite_or_nan(offroad_distance),
            "risk_field_lane_index": lane_index,
            "risk_field_lane_longitudinal": self._finite_or_nan(lane_longitudinal),
            "risk_field_lateral_offset": self._finite_or_nan(lateral_offset),
            "risk_field_lane_width": self._finite_or_nan(lane_width),
            "risk_field_dist_to_left_boundary": self._finite_or_nan(left_dist),
            "risk_field_dist_to_right_boundary": self._finite_or_nan(right_dist),
            "risk_field_side0_line_kind": side0_line_profile["kind"],
            "risk_field_side1_line_kind": side1_line_profile["kind"],
            "risk_field_side0_line_factor": float(side0_line_profile["factor"]),
            "risk_field_side1_line_factor": float(side1_line_profile["factor"]),
        }

    def _vehicle_risk(self, env: Any, vehicle: Any, lane: Any) -> Tuple[float, Dict[str, Any]]:
        """计算周围车辆相关风险（静态势场+动态势场+车头时距+TTC）
        
        Args:
            env: MetaDrive环境
            vehicle: 主车对象
            lane: 当前车道
            
        Returns:
            (vehicle_total_cost, info) 元组
        """
        max_distance = self._cfg("risk_field_max_distance")
        ego_pos = self._xy(getattr(vehicle, "position", (0.0, 0.0)))
        ego_length = self._dimension(vehicle, "top_down_length", 4.5)
        ego_width = self._dimension(vehicle, "top_down_width", 2.0)
        ref_heading = self._reference_heading(vehicle, lane)
        ego_forward_speed = self._forward_speed(vehicle, ref_heading)
        ego_lane_width = self._ego_lane_width(vehicle, lane)

        vehicle_cost = 0.0
        headway_cost = 0.0
        ttc_cost = 0.0
        count = 0
        nearest_distance = math.inf
        min_headway_time = math.inf
        min_ttc = math.inf

        # ========== 遍历周围车辆 ==========
        for other in self._iter_surrounding_vehicles(env, vehicle):
            other_pos = self._xy(getattr(other, "position", (0.0, 0.0)))
            distance = float(np.linalg.norm(other_pos - ego_pos))
            if distance > max_distance:
                continue

            count += 1
            nearest_distance = min(nearest_distance, distance)
            
            # 计算相对位置（Frenet坐标或笛卡尔坐标）
            delta_long, delta_lat = self._frame_delta(vehicle, other, lane, ref_heading)
            other_length = self._dimension(other, "top_down_length", 4.5)
            other_width = self._dimension(other, "top_down_width", 2.0)
            
            # 计算以障碍物为中心的相对坐标
            obs_long, obs_lat = self._oriented_delta(other, ego_pos, ref_heading)
            
            # ========== 计算车辆势场风险（静态+动态） ==========
            vehicle_cost += self._vehicle_potential_risk(
                obs_long,
                obs_lat,
                ego_forward_speed,
                other,
                ref_heading,
                other_length,
            )

            # ========== 计算车头时距风险（仅针对同车道前车） ==========
            is_front_vehicle = delta_long > 0.0
            is_same_corridor = abs(delta_lat) <= max(ego_lane_width * 0.75, (ego_width + other_width) / 2.0)
            if not (is_front_vehicle and is_same_corridor):
                continue

            # 计算纵向间隙
            front_gap = max(delta_long - (ego_length + other_length) / 2.0, 0.0)
            speed_for_headway = max(abs(ego_forward_speed), self._cfg("risk_field_min_speed"))
            headway_time = front_gap / speed_for_headway
            min_headway_time = min(min_headway_time, headway_time)
            
            # 对数惩罚：时距越小惩罚越大
            headway_cost += self._time_threshold_cost(
                headway_time,
                self._cfg("risk_field_headway_time_threshold"),
                self._cfg("risk_field_headway_cost_clip"),
            )

            # ========== 计算碰撞时间(TTC)风险 ==========
            other_forward_speed = self._forward_speed(other, ref_heading)
            closing_speed = ego_forward_speed - other_forward_speed
            if closing_speed > self.EPS:
                ttc = front_gap / closing_speed
                min_ttc = min(min_ttc, ttc)
                ttc_cost += self._time_threshold_cost(
                    ttc,
                    self._cfg("risk_field_ttc_threshold"),
                    self._cfg("risk_field_ttc_cost_clip"),
                )

        info = {
            "risk_field_vehicle_cost": float(vehicle_cost),
            "risk_field_headway_cost": float(headway_cost),
            "risk_field_ttc_cost": float(ttc_cost),
            "risk_field_surrounding_vehicle_count": int(count),
            "risk_field_nearest_vehicle_distance": self._finite_or_nan(nearest_distance),
            "risk_field_min_headway_time": self._finite_or_nan(min_headway_time),
            "risk_field_min_ttc": self._finite_or_nan(min_ttc),
        }
        return float(vehicle_cost + headway_cost + ttc_cost), info

    def _object_risk(self, env: Any, vehicle: Any, lane: Any) -> Tuple[float, Dict[str, Any]]:
        """计算静态障碍物风险（交通锥、路障等）
        
        Args:
            env: MetaDrive环境
            vehicle: 主车对象
            lane: 当前车道
            
        Returns:
            (object_cost, info) 元组
        """
        max_distance = self._cfg("risk_field_max_distance")
        ego_pos = self._xy(getattr(vehicle, "position", (0.0, 0.0)))
        ref_heading = self._reference_heading(vehicle, lane)

        object_cost = 0.0
        count = 0
        nearest_distance = math.inf

        # ========== 遍历静态障碍物 ==========
        for obj in self._iter_static_objects(env):
            obj_pos = self._xy(getattr(obj, "position", (0.0, 0.0)))
            distance = float(np.linalg.norm(obj_pos - ego_pos))
            if distance > max_distance:
                continue

            count += 1
            nearest_distance = min(nearest_distance, distance)
            
            # 计算相对位置（以障碍物为中心）
            delta_long, delta_lat = self._oriented_delta(obj, ego_pos, ref_heading)
            
            # 超髙斯风险模型
            object_cost += self._super_gaussian_risk(
                delta_long,
                delta_lat,
                self._cfg("risk_field_object_longitudinal_sigma"),
                self._cfg("risk_field_object_lateral_sigma"),
                self._cfg("risk_field_object_beta"),
            )

        return float(object_cost), {
            "risk_field_object_cost": float(object_cost),
            "risk_field_surrounding_object_count": int(count),
            "risk_field_nearest_object_distance": self._finite_or_nan(nearest_distance),
        }

    def _current_lane(self, env: Any, vehicle: Any) -> Any:
        """获取主车当前所在车道
        
        优先级：
        1. 如果使用最近车道模式，返回最近车道
        2. 从navigation.current_lane获取
        3. 从vehicle.lane获取
        4. 从路网查询最近车道
        
        Args:
            env: MetaDrive环境
            vehicle: 主车对象
            
        Returns:
            车道对象，如果找不到则返回None
        """
        if getattr(vehicle, "_risk_field_use_closest_lane", False):
            lane = self._closest_lane(env, getattr(vehicle, "position", None))
            if lane is not None:
                return lane

        navigation = getattr(vehicle, "navigation", None)
        lane = getattr(navigation, "current_lane", None)
        if lane is not None:
            return lane

        try:
            lane = getattr(vehicle, "lane", None)
        except Exception:
            lane = None
        if lane is not None:
            return lane

        road_network = self._road_network(env)
        if road_network is None:
            return None
        try:
            lane_index = self._extract_lane_index(road_network.get_closest_lane_index(vehicle.position))
            return road_network.get_lane(lane_index)
        except Exception:
            return None

    def _closest_lane(self, env: Any, position: Any) -> Any:
        """获取距离指定位置最近的车道
        
        Args:
            env: MetaDrive环境
            position: 位置坐标 [x, y]
            
        Returns:
            车道对象
        """
        road_network = self._road_network(env)
        if road_network is None or position is None:
            return None
        try:
            lane_index = self._extract_lane_index(road_network.get_closest_lane_index(position))
            return road_network.get_lane(lane_index)
        except Exception:
            return None

    def _road_network(self, env: Any) -> Any:
        """获取路网对象
        
        Args:
            env: MetaDrive环境
            
        Returns:
            road_network对象
        """
        current_map = getattr(env, "current_map", None)
        return getattr(current_map, "road_network", None)

    def _route_boundary_distances(self, vehicle: Any) -> Tuple[float, float]:
        """获取到左右边界的距离
        
        Args:
            vehicle: 车辆对象
            
        Returns:
            (left_distance, right_distance) 元组
        """
        left = getattr(vehicle, "dist_to_left_side", math.nan)
        right = getattr(vehicle, "dist_to_right_side", math.nan)
        if math.isfinite(self._safe_float(left)) and math.isfinite(self._safe_float(right)):
            return self._safe_float(left), self._safe_float(right)

        try:
            left, right = vehicle._dist_to_route_left_right()
            return self._safe_float(left), self._safe_float(right)
        except Exception:
            return math.nan, math.nan

    def _lane_boundary_fallback(self, lane: Any, longitudinal: float, lateral: float) -> Tuple[float, float]:
        """当无法获取边界距离时，用车道宽度估算
        
        Args:
            lane: 车道对象
            longitudinal: 纵向坐标
            lateral: 横向偏移
            
        Returns:
            (left_dist, right_dist) 估算值
        """
        if lane is None or not math.isfinite(lateral):
            return math.nan, math.nan
        width = self._lane_width(lane, longitudinal)
        if not math.isfinite(width):
            return math.nan, math.nan
        return lateral + width / 2.0, width / 2.0 - lateral

    def _lane_surface_state(self, env: Any, position: np.ndarray) -> Tuple[bool, float]:
        """检查位置是否在可行驶路面上
        
        Args:
            env: MetaDrive环境
            position: 待检查位置 [x, y]
            
        Returns:
            (on_road, offroad_distance) 元组：
            - on_road: 是否在道路上
            - offroad_distance: 偏离距离（如果在道路上则为0）
        """
        road_network = self._road_network(env)
        if road_network is None:
            return True, 0.0

        # 收集候选车道（最多12个最近车道）
        candidate_lanes = []
        try:
            closest = road_network.get_closest_lane_index(position, return_all=True)
            for _, lane_index in closest[:12]:
                candidate_lanes.append(road_network.get_lane(lane_index))
        except Exception:
            lane = self._closest_lane(env, position)
            if lane is not None:
                candidate_lanes.append(lane)

        margin = self._cfg("risk_field_on_lane_margin")
        min_surface_distance = math.inf
        
        # 检查每个候选车道
        for lane in candidate_lanes:
            longitudinal, lateral = self._lane_coordinates(lane, position)
            if not (math.isfinite(longitudinal) and math.isfinite(lateral)):
                continue
            length = self._safe_float(getattr(lane, "length", math.nan))
            width = self._lane_width(lane, longitudinal)
            if not (math.isfinite(length) and math.isfinite(width) and width > self.EPS):
                continue

            # 计算超出车道范围的距离
            longitudinal_over = max(-longitudinal, longitudinal - length, 0.0)
            lateral_over = max(abs(lateral) - width / 2.0, 0.0)
            surface_distance = float(math.hypot(longitudinal_over, lateral_over))
            min_surface_distance = min(min_surface_distance, surface_distance)
            
            # 如果在容差范围内，认为在道路上
            if surface_distance <= margin:
                return True, 0.0

        if not math.isfinite(min_surface_distance):
            min_surface_distance = 0.0
        return False, min_surface_distance

    def _frame_delta(self, ego: Any, obj: Any, lane: Any, ref_heading: np.ndarray) -> Tuple[float, float]:
        """计算物体相对于主车的位置差
        
        优先使用Frenet坐标系，如果失败则使用笛卡尔坐标系投影。
        
        Args:
            ego: 主车对象
            obj: 目标物体
            lane: 当前车道
            ref_heading: 参考航向向量
            
        Returns:
            (delta_longitudinal, delta_lateral) 相对位置
        """
        ego_pos = self._xy(getattr(ego, "position", (0.0, 0.0)))
        obj_pos = self._xy(getattr(obj, "position", (0.0, 0.0)))

        # 尝试使用Frenet坐标
        if lane is not None:
            ego_s, ego_l = self._lane_coordinates(lane, ego_pos)
            obj_s, obj_l = self._lane_coordinates(lane, obj_pos)
            if all(math.isfinite(v) for v in (ego_s, ego_l, obj_s, obj_l)):
                return float(obj_s - ego_s), float(obj_l - ego_l)

        # 回退到笛卡尔坐标投影
        delta = obj_pos - ego_pos
        lateral = np.array([-ref_heading[1], ref_heading[0]], dtype=float)
        return float(np.dot(delta, ref_heading)), float(np.dot(delta, lateral))

    def _oriented_delta(self, obj: Any, position: np.ndarray, fallback_heading: np.ndarray) -> Tuple[float, float]:
        """计算以物体自身朝向为基准的相对坐标
        
        Args:
            obj: 物体对象
            position: 参考位置（通常是主车位置）
            fallback_heading: 备用航向向量
            
        Returns:
            (longitudinal, lateral) 相对坐标
        """
        obj_pos = self._xy(getattr(obj, "position", (0.0, 0.0)))
        forward = self._object_heading(obj, fallback_heading)
        left = np.array([-forward[1], forward[0]], dtype=float)
        delta = position - obj_pos
        return float(np.dot(delta, forward)), float(np.dot(delta, left))

    def _object_heading(self, obj: Any, fallback_heading: np.ndarray) -> np.ndarray:
        """获取物体的航向单位向量
        
        Args:
            obj: 物体对象
            fallback_heading: 备用航向
            
        Returns:
            单位航向向量
        """
        heading = getattr(obj, "heading", None)
        if heading is not None:
            heading = self._xy(heading)
            norm = float(np.linalg.norm(heading))
            if norm > self.EPS:
                return heading / norm

        theta = self._safe_float(getattr(obj, "heading_theta", math.nan))
        if math.isfinite(theta):
            return np.array([math.cos(theta), math.sin(theta)], dtype=float)

        fallback_norm = float(np.linalg.norm(fallback_heading))
        if fallback_norm > self.EPS:
            return fallback_heading / fallback_norm
        return np.array([1.0, 0.0], dtype=float)

    def _reference_heading(self, vehicle: Any, lane: Any) -> np.ndarray:
        """获取参考航向向量
        
        优先级：
        1. 车道在该位置的切线方向
        2. 车辆当前航向
        3. 车辆航向角
        
        Args:
            vehicle: 车辆对象
            lane: 车道对象
            
        Returns:
            单位航向向量
        """
        if lane is not None:
            ego_s, _ = self._lane_coordinates(lane, self._xy(getattr(vehicle, "position", (0.0, 0.0))))
            if math.isfinite(ego_s):
                try:
                    theta = float(lane.heading_theta_at(ego_s))
                    return np.array([math.cos(theta), math.sin(theta)], dtype=float)
                except Exception:
                    pass

        heading = getattr(vehicle, "heading", None)
        if heading is not None:
            heading = self._xy(heading)
            norm = float(np.linalg.norm(heading))
            if norm > self.EPS:
                return heading / norm

        theta = self._safe_float(getattr(vehicle, "heading_theta", 0.0))
        return np.array([math.cos(theta), math.sin(theta)], dtype=float)

    def _forward_speed(self, obj: Any, ref_heading: np.ndarray) -> float:
        """计算物体在参考方向上的前向速度
        
        Args:
            obj: 物体对象
            ref_heading: 参考航向向量
            
        Returns:
            前向速度（m/s）
        """
        velocity = getattr(obj, "velocity", None)
        if velocity is None:
            return self._safe_float(getattr(obj, "speed", 0.0))
        return float(np.dot(self._xy(velocity), ref_heading))

    def _ego_lane_width(self, vehicle: Any, lane: Any) -> float:
        """获取主车所在车道的宽度
        
        Args:
            vehicle: 主车对象
            lane: 车道对象
            
        Returns:
            车道宽度（米），默认3.5米
        """
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None:
            try:
                width = float(navigation.get_current_lane_width())
                if math.isfinite(width) and width > self.EPS:
                    return width
            except Exception:
                pass
        if lane is not None:
            longitudinal, _ = self._lane_coordinates(lane, self._xy(getattr(vehicle, "position", (0.0, 0.0))))
            width = self._lane_width(lane, longitudinal)
            if math.isfinite(width) and width > self.EPS:
                return width
        return 3.5

    def _iter_surrounding_vehicles(self, env: Any, ego: Any) -> Iterable[Any]:
        """迭代环境中所有周围车辆（排除主车）
        
        从多个来源收集车辆：
        1. env.agents
        2. engine.get_objects
        3. traffic_manager
        4. agent_manager
        
        Args:
            env: MetaDrive环境
            ego: 主车对象
            
        Returns:
            周围车辆迭代器
        """
        engine = getattr(env, "engine", None)
        if engine is None:
            return []

        vehicles = {}
        self._collect_vehicle_candidates(vehicles, getattr(env, "agents", None))

        try:
            self._collect_vehicle_candidates(vehicles, engine.get_objects(self._is_vehicle_object).values())
        except Exception:
            pass

        # 从traffic_manager收集
        traffic_manager = getattr(engine, "traffic_manager", None)
        for attr in ("_traffic_vehicles", "traffic_vehicles", "block_triggered_vehicles"):
            candidates = getattr(traffic_manager, attr, None)
            if attr == "block_triggered_vehicles" and candidates is not None:
                flattened = []
                for block_vehicles in candidates:
                    flattened.extend(getattr(block_vehicles, "vehicles", []) or [])
                candidates = flattened
            self._collect_vehicle_candidates(vehicles, candidates)

        # 从agent_manager收集
        agent_manager = getattr(engine, "agent_manager", None)
        for attr in ("active_agents", "_active_objects", "_pending_objects", "_dying_objects"):
            candidates = getattr(agent_manager, attr, None)
            if attr == "_dying_objects" and isinstance(candidates, dict):
                candidates = [value[0] if isinstance(value, (list, tuple)) else value for value in candidates.values()]
            self._collect_vehicle_candidates(vehicles, candidates)

        # 过滤掉主车
        ego_name = getattr(ego, "name", None)
        ego_id = getattr(ego, "id", None)
        return [
            vehicle
            for vehicle in vehicles.values()
            if vehicle is not ego
            and getattr(vehicle, "name", None) != ego_name
            and getattr(vehicle, "id", None) != ego_id
        ]

    def _collect_vehicle_candidates(self, output: Dict[Any, Any], candidates: Any) -> None:
        """收集车辆候选对象到输出字典
        
        Args:
            output: 输出字典
            candidates: 候选对象（可以是单个对象、列表或字典）
        """
        if candidates is None:
            return
        if isinstance(candidates, dict):
            iterable = candidates.values()
        else:
            iterable = candidates
        try:
            iterator = iter(iterable)
        except TypeError:
            iterator = iter([iterable])
        for vehicle in iterator:
            if not self._is_vehicle_object(vehicle):
                continue
            key = getattr(vehicle, "name", None) or getattr(vehicle, "id", None) or id(vehicle)
            output[key] = vehicle

    def _iter_static_objects(self, env: Any) -> Iterable[Any]:
        """迭代环境中所有静态交通障碍物
        
        Args:
            env: MetaDrive环境
            
        Returns:
            静态障碍物迭代器
        """
        engine = getattr(env, "engine", None)
        if engine is None:
            return []

        try:
            return list(engine.get_objects(self._is_traffic_object).values())
        except Exception:
            return []

    @staticmethod
    def _is_vehicle_object(obj: Any) -> bool:
        """判断对象是否为车辆
        
        Args:
            obj: 待判断对象
            
        Returns:
            是否为车辆
        """
        if BaseVehicle is not None and isinstance(obj, BaseVehicle):
            return True
        return (
            hasattr(obj, "navigation")
            and hasattr(obj, "position")
            and hasattr(obj, "heading")
            and (hasattr(obj, "velocity") or hasattr(obj, "speed"))
        )

    @staticmethod
    def _is_traffic_object(obj: Any) -> bool:
        """判断对象是否为交通障碍物
        
        Args:
            obj: 待判断对象
            
        Returns:
            是否为交通障碍物
        """
        if TrafficObject is not None and isinstance(obj, TrafficObject):
            return True
        if RiskFieldCalculator._is_vehicle_object(obj):
            return False
        class_names = {cls.__name__ for cls in type(obj).mro()}
        return (
            "TrafficObject" in class_names
            or type(obj).__name__.startswith("Traffic")
        ) and hasattr(obj, "position")

    def _lane_coordinates(self, lane: Any, position: np.ndarray) -> Tuple[float, float]:
        """将世界坐标转换为Frenet坐标
        
        Args:
            lane: 车道对象
            position: 世界坐标 [x, y]
            
        Returns:
            (longitudinal, lateral) Frenet坐标
        """
        try:
            longitudinal, lateral = lane.local_coordinates(position)
            return self._safe_float(longitudinal), self._safe_float(lateral)
        except Exception:
            return math.nan, math.nan

    def _lane_width(self, lane: Any, longitudinal: float) -> float:
        """获取指定纵向位置的车道宽度
        
        Args:
            lane: 车道对象
            longitudinal: 纵向坐标
            
        Returns:
            车道宽度（米）
        """
        try:
            return self._safe_float(lane.width_at(longitudinal))
        except Exception:
            return self._safe_float(getattr(lane, "width", math.nan))

    def lane_line_risk_profile(self, lane: Any, side: int) -> Dict[str, Any]:
        """返回车道边线的风险语义配置。

        该方法供真实cost和topdown可视化共用，确保图中颜色与训练时风险语义一致。
        kind取值：
        - broken: 虚线，允许换道，低风险
        - solid: 普通实线，不鼓励跨越，中高风险
        - boundary: 道路边界/护栏，高风险
        - oncoming: 黄色/对向分隔线，最高风险
        - none: 无可用边线，不产生线风险
        """
        line_type, line_color = self._lane_line_type_color(lane, side)
        kind = self._lane_line_kind(line_type, line_color)
        sigma_key = {
            "broken": "risk_field_broken_line_sigma",
            "solid": "risk_field_lane_edge_sigma",
            "boundary": "risk_field_boundary_sigma",
            "oncoming": "risk_field_lane_edge_sigma",
        }.get(kind, "risk_field_lane_edge_sigma")
        factor_key = {
            "broken": "risk_field_broken_line_factor",
            "solid": "risk_field_solid_line_factor",
            "boundary": "risk_field_boundary_line_factor",
            "oncoming": "risk_field_oncoming_line_factor",
        }.get(kind)
        factor = 0.0 if factor_key is None else max(self._cfg(factor_key), 0.0)
        return {
            "kind": kind,
            "factor": float(factor),
            "sigma": max(self._cfg(sigma_key), self.EPS),
            "line_type": "" if line_type is None else str(line_type),
            "line_color": "" if line_color is None else str(line_color),
        }

    def lane_line_risk_components(self, distance: Any, profile: Dict[str, Any]) -> Dict[str, Any]:
        """计算车道线双层风险组件，支持标量和numpy数组。

        total = factor * ((1 - shoulder_weight) * core + shoulder_weight * shoulder)

        - core: 窄而强的超高斯核，只在线中心附近给出大惩罚
        - shoulder: 宽而弱的高斯肩部，提前提供低强度预警
        """
        distance_array = np.maximum(np.asarray(distance, dtype=float), 0.0)
        factor = max(self._safe_float(profile.get("factor", 0.0)), 0.0)
        base_sigma = max(self._safe_float(profile.get("sigma", self._cfg("risk_field_lane_edge_sigma"))), self.EPS)
        beta = max(self._cfg("risk_field_lane_beta"), self.EPS)
        core_sigma = max(base_sigma * self._cfg("risk_field_lane_core_sigma_scale"), self.EPS)
        shoulder_sigma = max(base_sigma * self._cfg("risk_field_lane_shoulder_sigma_scale"), self.EPS)
        shoulder_weight = min(max(self._cfg("risk_field_lane_shoulder_weight"), 0.0), 1.0)
        core_weight = 1.0 - shoulder_weight

        core_exponent = -((distance_array ** 2) / (core_sigma ** 2)) ** beta
        core_cost = np.exp(np.maximum(core_exponent, -80.0))
        shoulder_cost = np.exp(-(distance_array ** 2) / (2.0 * shoulder_sigma ** 2))
        total_cost = factor * (core_weight * core_cost + shoulder_weight * shoulder_cost)

        if core_cost.shape == ():
            return {
                "core": float(core_weight * factor * core_cost),
                "shoulder": float(shoulder_weight * factor * shoulder_cost),
                "total": float(total_cost),
            }
        return {
            "core": core_weight * factor * core_cost,
            "shoulder": shoulder_weight * factor * shoulder_cost,
            "total": total_cost,
        }

    @staticmethod
    def _lane_line_type_color(lane: Any, side: int) -> Tuple[Any, Any]:
        line_type = None
        line_color = None
        line_types = getattr(lane, "line_types", None)
        line_colors = getattr(lane, "line_colors", None)
        if line_types is not None:
            try:
                line_type = line_types[side]
            except Exception:
                line_type = None
        if line_colors is not None:
            try:
                line_color = line_colors[side]
            except Exception:
                line_color = None
        return line_type, line_color

    def _lane_line_kind(self, line_type: Any, line_color: Any) -> str:
        line_text = self._line_text(line_type)
        if self._is_line_type(line_type, "NONE") or "none" in line_text or "unknown" in line_text:
            return "none"

        if self._is_yellow_line(line_color) or "yellow" in line_text:
            return "oncoming"
        if (
            self._is_line_type(line_type, "SIDE")
            or self._is_line_type(line_type, "GUARDRAIL")
            or "boundary" in line_text
            or "guardrail" in line_text
            or "side" in line_text
        ):
            return "boundary"
        if self._is_line_type(line_type, "BROKEN") or "broken" in line_text:
            return "broken"
        if self._is_line_type(line_type, "CONTINUOUS") or "continuous" in line_text or "solid" in line_text:
            return "solid"
        return "solid"

    @staticmethod
    def _is_line_type(line_type: Any, name: str) -> bool:
        if PGLineType is None or line_type is None:
            return False
        expected = getattr(PGLineType, name, None)
        return expected is not None and line_type == expected

    @staticmethod
    def _line_text(line_type: Any) -> str:
        return "" if line_type is None else str(line_type).lower()

    @staticmethod
    def _is_yellow_line(line_color: Any) -> bool:
        if line_color is None:
            return False
        if PGLineColor is not None and line_color == getattr(PGLineColor, "YELLOW", None):
            return True
        if "yellow" in str(line_color).lower():
            return True
        try:
            rgb = np.asarray(line_color, dtype=float).reshape(-1)[:3]
        except Exception:
            return False
        if rgb.size < 3:
            return False
        if float(np.nanmax(rgb)) > 1.0:
            rgb = rgb / 255.0
        return bool(rgb[0] > 0.8 and rgb[1] > 0.55 and rgb[2] < 0.25)

    def _lane_index(self, lane: Any) -> str:
        """获取车道索引字符串
        
        Args:
            lane: 车道对象
            
        Returns:
            车道索引
        """
        index = getattr(lane, "index", None)
        return "" if index is None else str(index)

    @staticmethod
    def _extract_lane_index(closest_lane_result: Any) -> Any:
        """从get_closest_lane_index返回值中提取车道索引
        
        Args:
            closest_lane_result: get_closest_lane_index的返回值
            
        Returns:
            车道索引
        """
        if (
            isinstance(closest_lane_result, tuple)
            and len(closest_lane_result) == 2
            and isinstance(closest_lane_result[1], (float, int, np.floating, np.integer))
        ):
            return closest_lane_result[0]
        return closest_lane_result

    def _dimension(self, obj: Any, attr: str, default: float) -> float:
        """安全地获取物体的尺寸属性
        
        Args:
            obj: 物体对象
            attr: 属性名
            default: 默认值
            
        Returns:
            尺寸值
        """
        value = getattr(obj, attr, default)
        if callable(value):
            value = value()
        value = self._safe_float(value)
        return value if math.isfinite(value) and value > self.EPS else default

    def _cfg(self, key: str) -> float:
        """获取配置参数值
        
        Args:
            key: 参数键名
            
        Returns:
            参数值，如果无效则返回默认值
        """
        value = self._safe_float(self.config.get(key, self.DEFAULTS[key]))
        return value if math.isfinite(value) else self.DEFAULTS[key]

    @staticmethod
    def _safe_float(value: Any) -> float:
        """安全地将值转换为浮点数
        
        Args:
            value: 待转换的值
            
        Returns:
            浮点数，如果转换失败则返回NaN
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    @staticmethod
    def _xy(value: Any) -> np.ndarray:
        """将值转换为二维数组
        
        Args:
            value: 输入值（可以是标量、列表、数组等）
            
        Returns:
            二维数组 [x, y]
        """
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size >= 2:
            return array[:2]
        if array.size == 1:
            return np.array([array[0], 0.0], dtype=float)
        return np.array([0.0, 0.0], dtype=float)

    @staticmethod
    def _one_dimensional_risk(distance: float, sigma: float) -> float:
        """计算一维高斯风险
        
        公式：exp(-distance² / (2 * sigma²))
        
        Args:
            distance: 距离
            sigma: 标准差
            
        Returns:
            风险值 [0, 1]
        """
        sigma = max(float(sigma), RiskFieldCalculator.EPS)
        return float(math.exp(-(max(distance, 0.0) ** 2) / (2.0 * sigma ** 2)))

    @staticmethod
    def _super_gaussian_risk_1d(distance: float, sigma: float, beta: float) -> float:
        """计算一维超髙斯风险
        
        公式：exp(-(distance² / sigma²)^beta)
        
        当beta=1时为标准高斯，beta>1时曲线更陡峭，边缘惩罚更强。
        推荐范围：1.5~2.5（建议起点为2.0）
        
        Args:
            distance: 距离
            sigma: 标准差（控制影响范围）
            beta: 超髙斯指数（控制曲线形状）
            
        Returns:
            风险值 [0, 1]
        """
        sigma = max(float(sigma), RiskFieldCalculator.EPS)
        beta = max(float(beta), RiskFieldCalculator.EPS)
        # 使用平方后再求幂的方式，避免负数问题
        exponent = -((max(distance, 0.0) ** 2) / (sigma ** 2)) ** beta
        return float(math.exp(max(exponent, -80.0)))

    def _vehicle_potential_risk(
        self,
        longitudinal: float,
        lateral: float,
        ego_forward_speed: float,
        other: Any,
        ref_heading: np.ndarray,
        other_length: float,
    ) -> float:
        """计算车辆势场风险（静态+动态）
        
        静态部分：超髙斯分布建模车辆占据空间
        动态部分：考虑相对速度的非对称风险场
        
        Args:
            longitudinal: 纵向相对距离
            lateral: 横向相对距离
            ego_forward_speed: 主车前向速度
            other: 其他车辆对象
            ref_heading: 参考航向
            other_length: 其他车辆长度
            
        Returns:
            车辆风险值
        """
        other_forward_speed = self._forward_speed(other, ref_heading)
        components = self.vehicle_potential_components(
            longitudinal,
            lateral,
            ego_forward_speed,
            other_forward_speed,
            other_length,
        )
        return float(components["total"])

    def vehicle_potential_components(
        self,
        longitudinal: Any,
        lateral: Any,
        ego_forward_speed: float,
        other_forward_speed: float,
        other_length: float,
    ) -> Dict[str, Any]:
        """计算车辆静态、动态和总势场组件，支持标量和numpy数组。

        该函数是车辆风险cost和调试可视化共用的公式入口，避免两边公式漂移。
        """
        longitudinal_array = np.asarray(longitudinal, dtype=float)
        lateral_array = np.asarray(lateral, dtype=float)

        sigma_long = max(self._cfg("risk_field_vehicle_longitudinal_sigma"), self.EPS)
        sigma_lat = max(self._cfg("risk_field_vehicle_lateral_sigma"), self.EPS)
        beta = max(self._cfg("risk_field_vehicle_beta"), self.EPS)
        static_exponent = -(
            ((longitudinal_array ** 2) / (sigma_long ** 2)) ** beta
            + ((lateral_array ** 2) / (sigma_lat ** 2)) ** beta
        )
        static_cost = np.exp(np.maximum(static_exponent, -80.0))

        speed_delta = abs(float(other_forward_speed) - float(ego_forward_speed))
        dynamic_sigma = max(
            self._cfg("risk_field_vehicle_dynamic_sigma_scale") * speed_delta,
            self._cfg("risk_field_vehicle_min_dynamic_sigma"),
        )
        lateral_sigma = max(self._cfg("risk_field_vehicle_lateral_sigma"), self.EPS)
        dynamic_exponent = -(
            (longitudinal_array ** 2) / (dynamic_sigma ** 2)
            + (lateral_array ** 2) / (lateral_sigma ** 2)
        )
        dynamic_cost = np.exp(np.maximum(dynamic_exponent, -80.0))

        relv = 1.0 if other_forward_speed >= ego_forward_speed else -1.0
        alpha = self._cfg("risk_field_vehicle_dynamic_alpha")
        sigmoid_arg = -relv * (longitudinal_array - alpha * other_length * relv)
        dynamic_cost = dynamic_cost / (1.0 + np.exp(np.clip(sigmoid_arg, -60.0, 60.0)))
        total_cost = static_cost + dynamic_cost

        if static_cost.shape == ():
            return {
                "static": float(static_cost),
                "dynamic": float(dynamic_cost),
                "total": float(total_cost),
            }
        return {
            "static": static_cost,
            "dynamic": dynamic_cost,
            "total": total_cost,
        }

    @staticmethod
    def _super_gaussian_risk(longitudinal: float, lateral: float, sigma_long: float, sigma_lat: float, beta: float) -> float:
        """计算超髙斯风险
        
        公式：exp(-((long²/sigma_long²)^beta + (lat²/sigma_lat²)^beta))
        
        当beta=1时为标准高斯，beta>1时更尖锐。
        
        Args:
            longitudinal: 纵向距离
            lateral: 横向距离
            sigma_long: 纵向标准差
            sigma_lat: 横向标准差
            beta: 超髙斯指数
            
        Returns:
            风险值 [0, 1]
        """
        sigma_long = max(float(sigma_long), RiskFieldCalculator.EPS)
        sigma_lat = max(float(sigma_lat), RiskFieldCalculator.EPS)
        beta = max(float(beta), RiskFieldCalculator.EPS)
        exponent = -(
            ((longitudinal ** 2) / (sigma_long ** 2)) ** beta
            + ((lateral ** 2) / (sigma_lat ** 2)) ** beta
        )
        return float(math.exp(max(exponent, -80.0)))

    @staticmethod
    def _time_threshold_cost(time_value: float, threshold: float, cost_clip: float) -> float:
        """计算时间阈值成本（用于Headway和TTC）
        
        公式：-ln(time_value / threshold)，当time_value < threshold时
        
        Args:
            time_value: 时间值（如车头时距或TTC）
            threshold: 阈值
            cost_clip: 成本上限
            
        Returns:
            成本值 [0, cost_clip]
        """
        if not math.isfinite(time_value) or threshold <= RiskFieldCalculator.EPS:
            return 0.0
        if time_value >= threshold:
            return 0.0
        normalized = max(time_value / threshold, RiskFieldCalculator.EPS)
        return RiskFieldCalculator._clip_nonnegative(-math.log(normalized), cost_clip)

    @staticmethod
    def _clip_nonnegative(value: float, upper: float) -> float:
        """将值限制在非负范围并裁剪到上限
        
        Args:
            value: 输入值
            upper: 上限
            
        Returns:
            裁剪后的值
        """
        value = max(float(value), 0.0)
        if upper is None or not math.isfinite(float(upper)):
            return value
        return min(value, max(float(upper), 0.0))

    @staticmethod
    def _finite_or_nan(value: float) -> float:
        """如果值有限则返回，否则返回NaN
        
        Args:
            value: 输入值
            
        Returns:
            有限值或NaN
        """
        value = RiskFieldCalculator._safe_float(value)
        return value if math.isfinite(value) else math.nan


class _RiskFieldSampleVehicle:
    """用于BEV热力图采样的轻量级车辆代理
    
    该类创建一个虚拟车辆对象，保留源车辆的尺寸和速度，
    但位置可以任意设置，用于风险场的网格采样。
    """

    def __init__(self, source_vehicle: Any, position: Any, *, speed: Optional[float], use_closest_lane: bool):
        """初始化采样车辆
        
        Args:
            source_vehicle: 源车辆（提供尺寸和速度参考）
            position: 采样位置
            speed: 可选的速度覆盖值
            use_closest_lane: 是否使用最近车道
        """
        self.name = getattr(source_vehicle, "name", "ego")
        self.position = RiskFieldCalculator._xy(position)
        self.navigation = getattr(source_vehicle, "navigation", None)
        self.heading = self._unit_heading(source_vehicle)
        self.heading_theta = RiskFieldCalculator._safe_float(getattr(source_vehicle, "heading_theta", 0.0))
        self.speed = self._source_speed(source_vehicle) if speed is None else float(speed)
        self.velocity = self.heading * self.speed
        self.top_down_length = self._dimension(source_vehicle, "top_down_length", 4.5)
        self.top_down_width = self._dimension(source_vehicle, "top_down_width", 2.0)
        self._risk_field_use_closest_lane = use_closest_lane

    @staticmethod
    def _source_speed(source_vehicle: Any) -> float:
        """获取源车辆速度
        
        Args:
            source_vehicle: 源车辆
            
        Returns:
            速度值
        """
        speed = RiskFieldCalculator._safe_float(getattr(source_vehicle, "speed", 0.0))
        return speed if math.isfinite(speed) else 0.0

    @staticmethod
    def _unit_heading(source_vehicle: Any) -> np.ndarray:
        """获取源车辆的单位航向向量
        
        Args:
            source_vehicle: 源车辆
            
        Returns:
            单位航向向量
        """
        heading = getattr(source_vehicle, "heading", None)
        if heading is not None:
            heading = RiskFieldCalculator._xy(heading)
            norm = float(np.linalg.norm(heading))
            if norm > RiskFieldCalculator.EPS:
                return heading / norm
        heading_theta = RiskFieldCalculator._safe_float(getattr(source_vehicle, "heading_theta", 0.0))
        return np.array([math.cos(heading_theta), math.sin(heading_theta)], dtype=float)

    @staticmethod
    def _dimension(source_vehicle: Any, attr: str, default: float) -> float:
        """获取源车辆的尺寸属性
        
        Args:
            source_vehicle: 源车辆
            attr: 属性名
            default: 默认值
            
        Returns:
            尺寸值
        """
        value = getattr(source_vehicle, attr, default)
        if callable(value):
            value = value()
        value = RiskFieldCalculator._safe_float(value)
        return value if math.isfinite(value) and value > RiskFieldCalculator.EPS else default
