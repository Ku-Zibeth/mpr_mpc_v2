"""MetaDrive风险场BEV（鸟瞰图）可视化模块

功能说明：
    该模块提供风险场的实时可视化功能，将风险值渲染为热力图，并在其上叠加道路、车辆、障碍物等元素。
    支持多种风险组件的独立显示（总风险、边界风险、车道风险、车辆风险等）。
    
主要特性：
    - 支持主车局部坐标系和世界坐标系两种显示模式
    - 自动过滤低风险的透明区域，突出高风险区域
    - 根据MetaDrive真实车道线类型绘制（虚线/实线/边界）
    - 动态标注周围车辆和障碍物的速度信息
    - 左上角实时显示当前风险分解统计
"""

import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .risk_field import RiskFieldCalculator


# 风险场组件名称到数据键名的映射表
COMPONENT_KEYS = {
    "total": "risk_field_cost",              # 总风险成本
    "risk": "risk_field_cost",               # 风险（别名）
    "road": "risk_field_road_cost",          # 道路风险（边界+车道）
    "boundary": "risk_field_boundary_cost",  # 道路边界风险
    "lane": "risk_field_lane_cost",          # 车道偏离风险
    "offroad": "risk_field_offroad_cost",    # 偏离道路风险
    "vehicle": "risk_field_vehicle_cost",    # 周围车辆风险
    "object": "risk_field_object_cost",      # 静态障碍物风险
    "headway": "risk_field_headway_cost",    # 车头时距风险
    "ttc": "risk_field_ttc_cost",            # 碰撞时间风险
}


def plot_risk_field_bev(
    env: Any,
    vehicle: Optional[Any] = None,
    *,
    calculator: Optional[RiskFieldCalculator] = None,
    save_path: Optional[str] = None,
    component: str = "total",
    front_range: Tuple[float, float] = (-20.0, 70.0),
    lateral_range: Tuple[float, float] = (-20.0, 20.0),
    resolution: float = 0.5,
    ego_frame: bool = True,
    lane_sample_step: float = 2.0,
    draw_lane_centers: bool = False,
    draw_lane_surfaces: bool = False,
    figsize: Tuple[float, float] = (11.0, 8.0),
    dpi: int = 120,
    cmap: str = "risk_legacy",
    vmax: Optional[float] = 1.2,
    risk_min_visible: float = 1e-4,
    interpolation: str = "bilinear",
    show: bool = True,
    annotate: bool = True,
):
    """绘制风险场BEV热力图，包含道路、主车、周围车辆和障碍物
    
    Args:
        env: 运行中的MetaDrive环境实例
        vehicle: 主车对象。如果省略，则使用env.agents中的第一辆车
        calculator: 可选的风险场计算器实例。默认从env.config创建
        save_path: 如果设置，将图形保存到此路径
        component: 要显示的风险组件类型，可选值：
                   total/road/boundary/lane/offroad/vehicle/object/headway/ttc
        front_range: 前方采样范围（米）。在主车坐标系中表示前后距离，
                     在世界坐标系中表示x轴偏移范围
        lateral_range: 横向采样范围（米）。在主车坐标系中表示左右距离，
                       在世界坐标系中表示y轴偏移范围
        resolution: 网格分辨率（米/像素），控制热力图的精细程度
        ego_frame: 如果为True，主车航向朝上显示；否则使用世界坐标系
        lane_sample_step: 车道中心线/边缘线的采样步长（米）
        draw_lane_centers: 是否绘制车道中心线。默认禁用，因为中心线不是真实的道路标线
        draw_lane_surfaces: 是否填充半透明的车道表面多边形，默认关闭避免路网整片铺满
        figsize: matplotlib图形尺寸（英寸）
        dpi: 保存图片时使用的DPI，越高越清晰但越慢
        cmap: 颜色映射方案。默认"risk_legacy"接近原版蓝-绿-黄-红，高风险为红色
        vmax: 颜色归一化的固定最大值。默认1.2对齐原版势场配色，设为None则自动调整
        risk_min_visible: 低于此阈值的风险值将被渲染为透明（过滤噪声）
        interpolation: 热力图插值方式。默认bilinear，让势场更接近连续场
        show: 是否显示matplotlib窗口
        annotate: 是否在车辆/障碍物上标注名称和速度
        
    Returns:
        (fig, ax, data) 三元组，其中data包含网格坐标和采样的风险值
    """

    import matplotlib.pyplot as plt

    # 获取主车对象
    vehicle = vehicle or _default_vehicle(env)
    if vehicle is None:
        raise ValueError("未找到主车。请显式传递vehicle参数或先重置环境。")

    # 初始化风险场计算器
    calculator = calculator or RiskFieldCalculator(getattr(env, "config", {}))
    
    # ========== 采样风险场数据 ==========
    data = sample_risk_field_bev(
        env,
        vehicle,
        calculator=calculator,
        component=component,
        front_range=front_range,
        lateral_range=lateral_range,
        resolution=resolution,
        ego_frame=ego_frame,
    )

    # ========== 创建matplotlib图形 ==========
    fig, ax = plt.subplots(figsize=figsize)
    
    # 掩码处理：将低于阈值的风险值设为透明
    display_risk = np.ma.masked_where(data["risk"] <= risk_min_visible, data["risk"])
    
    # 绘制风险热力图
    image = ax.imshow(
        display_risk,
        origin="lower",                      # 原点在左下角
        extent=data["extent"],               # 坐标轴范围
        cmap=_resolve_cmap(cmap),            # 颜色映射
        vmin=0.0,                            # 最小值固定为0
        vmax=_resolve_vmax(data["risk"], vmax, risk_min_visible),  # 最大值
        aspect="equal",                      # 保持纵横比
        interpolation=interpolation,          # 平滑插值，减少像素块感
        zorder=1,                            # 图层顺序：热力图在最底层
    )
    
    # 添加颜色条
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=data["component_key"])

    # ========== 绘制道路和物体 ==========
    transform = data["world_to_plot"]        # 坐标转换函数
    plot_bounds = data["plot_bounds"]        # 绘图边界
    
    # 锁定坐标轴范围，防止被远处物体拉伸
    ax.set_xlim(plot_bounds[0], plot_bounds[1])
    ax.set_ylim(plot_bounds[2], plot_bounds[3])
    
    # 绘制车道线
    _draw_lanes(
        ax,
        env,
        transform,
        plot_bounds,
        lane_sample_step,
        draw_centers=draw_lane_centers,
        draw_surfaces=draw_lane_surfaces,
    )
    
    # 绘制周围车辆和障碍物
    _draw_objects(ax, env, vehicle, calculator, transform, plot_bounds, annotate=annotate)
    
    # 绘制当前风险统计框
    _draw_current_risk_box(ax, env, vehicle, calculator)

    # ========== 设置坐标轴标签和标题 ==========
    if ego_frame:
        ax.set_xlabel("lateral left [m]")
        ax.set_ylabel("forward [m]")
        ax.set_title(f"Risk field BEV ({data['component_key']}, ego heading up)")
    else:
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.set_title(f"Risk field BEV ({data['component_key']}, world frame)")

    # 添加白色网格线
    ax.grid(color="white", alpha=0.12, linewidth=0.6)
    
    # 再次锁定坐标轴范围（确保不被自动调整）
    ax.set_xlim(plot_bounds[0], plot_bounds[1])
    ax.set_ylim(plot_bounds[2], plot_bounds[3])
    
    # 添加图例
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()

    # ========== 保存或显示图形 ==========
    if save_path:
        directory = os.path.dirname(os.path.abspath(save_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    return fig, ax, data


def sample_risk_field_bev(
    env: Any,
    vehicle: Any,
    *,
    calculator: Optional[RiskFieldCalculator] = None,
    component: str = "total",
    front_range: Tuple[float, float] = (-20.0, 70.0),
    lateral_range: Tuple[float, float] = (-20.0, 20.0),
    resolution: float = 1.0,
    ego_frame: bool = True,
) -> Dict[str, Any]:
    """在BEV网格上采样风险场数据（不绘制图形）
    
    该函数遍历指定范围内的每个网格点，调用风险场计算器获取该位置的风险值，
    最终生成二维风险矩阵用于可视化。
    
    Args:
        env: MetaDrive环境实例
        vehicle: 主车对象
        calculator: 风险场计算器（可选）
        component: 风险组件类型
        front_range: 前方采样范围（米）
        lateral_range: 横向采样范围（米）
        resolution: 网格分辨率（米/像素）
        ego_frame: 是否使用主车局部坐标系
        
    Returns:
        字典，包含以下键：
        - risk: 二维风险数组 (n_front, n_lateral)
        - extent: 坐标轴范围 [left, right, bottom, top]
        - plot_bounds: 绘图边界 (x_min, x_max, y_min, y_max)
        - component_key: 风险组件的数据键名
        - front_values: 前方采样点数组
        - lateral_values: 横向采样点数组
        - world_to_plot: 世界坐标到绘图坐标的转换函数
    """

    if resolution <= 0:
        raise ValueError("分辨率必须为正数。")

    # 初始化计算器
    calculator = calculator or RiskFieldCalculator(getattr(env, "config", {}))
    component_key = _component_key(component)
    
    # 获取主车位置和朝向
    ego_pos = RiskFieldCalculator._xy(getattr(vehicle, "position", (0.0, 0.0)))
    forward_vec = _unit_heading(vehicle)                    # 前进方向单位向量
    left_vec = np.array([-forward_vec[1], forward_vec[0]], dtype=float)  # 左侧方向单位向量

    # 生成采样点数组
    front_values = _grid_values(front_range, resolution)
    lateral_values = _grid_values(lateral_range, resolution)
    risk = np.zeros((len(front_values), len(lateral_values)), dtype=float)

    # ========== 逐点采样风险值 ==========
    if ego_frame:
        # 主车局部坐标系：以主车为原点，航向为y轴正方向
        for row, forward in enumerate(front_values):
            for col, lateral in enumerate(lateral_values):
                # 计算世界坐标
                world_position = ego_pos + forward_vec * forward + left_vec * lateral
                # 查询该位置的风险值
                cost, info = calculator.calculate_at_position(env, vehicle, world_position)
                risk[row, col] = float(info.get(component_key, cost))

        # 设置坐标轴范围
        extent = [lateral_values[0], lateral_values[-1], front_values[0], front_values[-1]]
        plot_bounds = (extent[0], extent[1], extent[2], extent[3])

        # 定义坐标转换函数：世界坐标 -> 绘图坐标
        def world_to_plot(point):
            delta = RiskFieldCalculator._xy(point) - ego_pos
            return np.array([np.dot(delta, left_vec), np.dot(delta, forward_vec)], dtype=float)

    else:
        # 世界坐标系：直接使用全局坐标
        x_values = ego_pos[0] + front_values
        y_values = ego_pos[1] + lateral_values
        for row, y in enumerate(y_values):
            for col, x in enumerate(x_values):
                world_position = np.array([x, y], dtype=float)
                cost, info = calculator.calculate_at_position(env, vehicle, world_position)
                risk[row, col] = float(info.get(component_key, cost))

        extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
        plot_bounds = (extent[0], extent[1], extent[2], extent[3])

        def world_to_plot(point):
            return RiskFieldCalculator._xy(point)

    return {
        "risk": risk,
        "extent": extent,
        "plot_bounds": plot_bounds,
        "component_key": component_key,
        "front_values": front_values,
        "lateral_values": lateral_values,
        "world_to_plot": world_to_plot,
    }


def resolve_map_bbox(env: Any, *, padding_m: float = 8.0) -> Tuple[float, float, float, float]:
    """Return the padded world-space bounding box of the current road network."""
    road_network = getattr(getattr(env, "current_map", None), "road_network", None)
    if road_network is None:
        raise ValueError("Current map has no road_network; cannot derive global bbox.")
    x_min, x_max, y_min, y_max = [float(v) for v in road_network.get_bounding_box()]
    padding = max(float(padding_m), 0.0)
    return (x_min - padding, x_max + padding, y_min - padding, y_max + padding)


def resolve_effective_resolution(
    bounds: Tuple[float, float, float, float],
    requested_resolution: float,
    *,
    max_grid_points: int = 180000,
) -> Tuple[float, Tuple[int, int]]:
    """Increase resolution when needed so the global sampling grid stays bounded."""
    resolution = max(float(requested_resolution), 1e-6)
    x_min, x_max, y_min, y_max = [float(v) for v in bounds]
    max_points = max(int(max_grid_points), 1)

    def _shape_at(res: float) -> Tuple[int, int]:
        x_count = int(len(_grid_values((x_min, x_max), res)))
        y_count = int(len(_grid_values((y_min, y_max), res)))
        return y_count, x_count

    grid_shape = _shape_at(resolution)
    point_count = int(grid_shape[0] * grid_shape[1])
    if point_count <= max_points:
        return resolution, grid_shape

    scale = math.sqrt(point_count / float(max_points))
    resolution *= max(scale, 1.0)
    grid_shape = _shape_at(resolution)
    while int(grid_shape[0] * grid_shape[1]) > max_points:
        resolution *= 1.02
        grid_shape = _shape_at(resolution)
    return float(resolution), grid_shape


def build_surface_road_mask(env: Any, surface: Any, *, margin_px: int = 3) -> np.ndarray:
    """Rasterize all lane polygons onto the given topdown surface."""
    from PIL import Image, ImageDraw, ImageFilter

    width = int(surface.get_width())
    height = int(surface.get_height())
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)

    for lane in _iter_lanes(env):
        polygon = []
        for point in getattr(lane, "polygon", []):
            world = RiskFieldCalculator._xy(point)
            if np.isfinite(world).all():
                polygon.append(surface.pos2pix(float(world[0]), float(world[1])))
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=255)

    margin = max(int(margin_px), 0)
    if margin > 0:
        kernel = max(3, margin * 2 + 1)
        mask_image = mask_image.filter(ImageFilter.MaxFilter(kernel))
    return np.asarray(mask_image, dtype=np.uint8) > 0


def sample_surface_mask_for_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    surface: Any,
    pixel_mask: np.ndarray,
) -> np.ndarray:
    """Project a world-space grid onto a pixel-space road mask."""
    pixel_mask = np.asarray(pixel_mask, dtype=bool)
    height, width = pixel_mask.shape
    grid_mask = np.zeros((len(y_values), len(x_values)), dtype=bool)

    for row, y_value in enumerate(np.asarray(y_values, dtype=float)):
        for col, x_value in enumerate(np.asarray(x_values, dtype=float)):
            pixel_x, pixel_y = surface.pos2pix(float(x_value), float(y_value))
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                grid_mask[row, col] = bool(pixel_mask[pixel_y, pixel_x])

    return grid_mask


def sample_risk_field_world_bbox(
    env: Any,
    vehicle: Any,
    *,
    calculator: Optional[RiskFieldCalculator] = None,
    component: str = "total",
    bbox: Optional[Tuple[float, float, float, float]] = None,
    padding_m: float = 8.0,
    resolution: float = 1.5,
    max_grid_points: int = 180000,
    road_grid_mask: Optional[np.ndarray] = None,
    surface: Optional[Any] = None,
    road_pixel_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Sample risk over a global world-space bbox instead of an ego-local window."""
    calculator = calculator or RiskFieldCalculator(getattr(env, "config", {}))
    component_key = _component_key(component)
    resolved_bbox = resolve_map_bbox(env, padding_m=padding_m) if bbox is None else tuple(float(v) for v in bbox)
    effective_resolution, grid_shape = resolve_effective_resolution(
        resolved_bbox,
        requested_resolution=resolution,
        max_grid_points=max_grid_points,
    )
    x_min, x_max, y_min, y_max = resolved_bbox
    x_values = _grid_values((x_min, x_max), effective_resolution)
    y_values = _grid_values((y_min, y_max), effective_resolution)
    risk = np.full((len(y_values), len(x_values)), np.nan, dtype=float)

    if road_grid_mask is None and surface is not None and road_pixel_mask is not None:
        road_grid_mask = sample_surface_mask_for_grid(x_values, y_values, surface, road_pixel_mask)

    if road_grid_mask is not None:
        road_grid_mask = np.asarray(road_grid_mask, dtype=bool)
        if road_grid_mask.shape != risk.shape:
            raise ValueError(
                "road_grid_mask shape mismatch: expected {}, got {}".format(risk.shape, road_grid_mask.shape)
            )

    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            if road_grid_mask is not None and not bool(road_grid_mask[row, col]):
                continue
            world_position = np.array([float(x_value), float(y_value)], dtype=float)
            cost, info = calculator.calculate_at_position(env, vehicle, world_position)
            risk[row, col] = float(info.get(component_key, cost))

    extent = [float(x_values[0]), float(x_values[-1]), float(y_values[0]), float(y_values[-1])]
    plot_bounds = (extent[0], extent[1], extent[2], extent[3])

    def world_to_plot(point):
        return RiskFieldCalculator._xy(point)

    return {
        "risk": risk,
        "extent": extent,
        "plot_bounds": plot_bounds,
        "component_key": component_key,
        "x_values": x_values,
        "y_values": y_values,
        "lateral_values": x_values,
        "front_values": y_values,
        "world_to_plot": world_to_plot,
        "map_bbox": resolved_bbox,
        "effective_resolution": float(effective_resolution),
        "requested_resolution": float(resolution),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "max_grid_points": int(max_grid_points),
        "road_grid_mask": road_grid_mask,
    }


def build_risk_surface_data(
    sample: Dict[str, Any],
    *,
    risk_min_visible: float = 1e-4,
    z_max: Optional[float] = None,
) -> Dict[str, Any]:
    """将二维风险采样结果转换为 3D 曲面绘制所需的网格数据。

    Args:
        sample: ``sample_risk_field_bev`` 返回的字典，至少包含 ``risk``、
            ``front_values`` 与 ``lateral_values``。
        risk_min_visible: 低于该阈值的风险值会被 mask，避免 3D 底面整体抬起。
        z_max: 可选的固定 z 轴上限；若未设置或 <= 0，则根据当前风险场自动估计。

    Returns:
        包含 ``x_grid``、``y_grid``、``z_grid``、``z_masked``、``max_risk``、
        ``z_max`` 等字段的新字典，同时保留输入采样结果中的原始键。
    """
    risk = np.asarray(sample["risk"], dtype=float)
    x_values_raw = sample["x_values"] if "x_values" in sample else sample["lateral_values"]
    y_values_raw = sample["y_values"] if "y_values" in sample else sample["front_values"]
    x_values = np.asarray(x_values_raw, dtype=float)
    y_values = np.asarray(y_values_raw, dtype=float)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    threshold = max(float(risk_min_visible), 0.0)
    visible_mask = np.isfinite(risk) & (risk > threshold)
    z_masked = np.ma.masked_where(~visible_mask, risk)

    finite_risk = risk[np.isfinite(risk)]
    max_risk = float(np.max(finite_risk)) if finite_risk.size else 0.0

    requested_z_max = None if z_max is None else float(z_max)
    if requested_z_max is not None and math.isfinite(requested_z_max) and requested_z_max > 0.0:
        resolved_z_max = requested_z_max
    else:
        visible_risk = risk[visible_mask]
        if visible_risk.size:
            resolved_z_max = max(float(np.max(visible_risk)) * 1.05, threshold * 10.0, 1e-3)
        else:
            resolved_z_max = max(max_risk * 1.05, threshold * 10.0, 1e-3)

    surface = dict(sample)
    surface.update(
        {
            "x_values": x_values,
            "y_values": y_values,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "z_grid": risk,
            "z_masked": z_masked,
            "visible_mask": visible_mask,
            "risk_min_visible": threshold,
            "max_risk": max_risk,
            "z_max": float(resolved_z_max),
        }
    )
    return surface


def sample_risk_surface_bev(
    env: Any,
    vehicle: Any,
    *,
    calculator: Optional[RiskFieldCalculator] = None,
    component: str = "total",
    front_range: Tuple[float, float] = (-20.0, 70.0),
    lateral_range: Tuple[float, float] = (-20.0, 20.0),
    resolution: float = 1.0,
    ego_frame: bool = True,
    risk_min_visible: float = 1e-4,
    z_max: Optional[float] = None,
) -> Dict[str, Any]:
    """采样风险场并返回适合 3D 曲面绘制的网格数据。"""
    sample = sample_risk_field_bev(
        env,
        vehicle,
        calculator=calculator,
        component=component,
        front_range=front_range,
        lateral_range=lateral_range,
        resolution=resolution,
        ego_frame=ego_frame,
    )
    return build_risk_surface_data(sample, risk_min_visible=risk_min_visible, z_max=z_max)


def sample_risk_surface_world_bbox(
    env: Any,
    vehicle: Any,
    *,
    calculator: Optional[RiskFieldCalculator] = None,
    component: str = "total",
    bbox: Optional[Tuple[float, float, float, float]] = None,
    padding_m: float = 8.0,
    resolution: float = 1.5,
    max_grid_points: int = 180000,
    road_grid_mask: Optional[np.ndarray] = None,
    surface: Optional[Any] = None,
    road_pixel_mask: Optional[np.ndarray] = None,
    risk_min_visible: float = 1e-4,
    z_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Sample a global world-space risk surface over the whole road-network bbox."""
    sample = sample_risk_field_world_bbox(
        env,
        vehicle,
        calculator=calculator,
        component=component,
        bbox=bbox,
        padding_m=padding_m,
        resolution=resolution,
        max_grid_points=max_grid_points,
        road_grid_mask=road_grid_mask,
        surface=surface,
        road_pixel_mask=road_pixel_mask,
    )
    return build_risk_surface_data(sample, risk_min_visible=risk_min_visible, z_max=z_max)


def _component_key(component: str) -> str:
    """将组件名称转换为数据键名
    
    Args:
        component: 组件名称（如"total"、"vehicle"等）
        
    Returns:
        对应的数据键名（如"risk_field_cost"）
    """
    component = component.strip().lower()
    return COMPONENT_KEYS.get(component, component)


def resolve_risk_cmap(cmap: str):
    """公开风险色带解析，便于论文图和测试复用同一色彩语义。"""
    return _resolve_cmap(cmap)


def _resolve_cmap(cmap: str):
    """解析颜色映射方案
    
    Args:
        cmap: 颜色映射名称
        
    Returns:
        matplotlib colormap对象
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if cmap == "risk_legacy":
        # 接近原版势场图的色系：低风险透明，随后蓝/青/黄/橙/红逐级增强。
        cmap_obj = LinearSegmentedColormap.from_list(
            "risk_legacy",
            [
                (1.0, 1.0, 1.0, 0.0),       # 透明白色（无风险）
                (0.12, 0.25, 0.85, 0.35),   # 蓝色
                (0.00, 0.78, 0.95, 0.55),   # 青色
                (0.30, 0.90, 0.28, 0.70),   # 绿色
                (1.00, 0.88, 0.12, 0.88),   # 黄色
                (1.00, 0.45, 0.10, 0.96),   # 橙色
                (0.80, 0.00, 0.00, 1.0),    # 红色（高风险）
            ],
        )
    elif cmap == "risk_red":
        # 自定义风险红色映射：透明 -> 浅橙 -> 深红 -> 暗红
        cmap_obj = LinearSegmentedColormap.from_list(
            "risk_red",
            [
                (1.0, 1.0, 1.0, 0.0),       # 透明白色（低风险）
                (1.0, 0.92, 0.86, 0.45),    # 浅橙色
                (1.0, 0.45, 0.25, 0.92),    # 深橙色
                (0.95, 0.04, 0.02, 1.0),    # 鲜红色
                (0.45, 0.00, 0.00, 1.0),    # 暗红色（高风险）
            ],
        )
    else:
        # 使用matplotlib内置的颜色映射
        cmap_obj = plt.get_cmap(cmap).copy()
    
    # 设置无效值（masked）为透明
    cmap_obj.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap_obj


def _resolve_vmax(risk: np.ndarray, fixed_vmax: Optional[float], min_visible: float) -> float:
    """确定颜色映射的最大值
    
    Args:
        risk: 风险值数组
        fixed_vmax: 用户指定的固定最大值
        min_visible: 最小可见阈值
        
    Returns:
        颜色映射的最大值
    """
    # 如果用户指定了固定值，优先使用
    if fixed_vmax is not None and math.isfinite(float(fixed_vmax)) and fixed_vmax > 0:
        return float(fixed_vmax)
    
    # 否则自动计算：取98%分位数，确保大部分区域有颜色变化
    finite_positive = risk[np.isfinite(risk) & (risk > min_visible)]
    if finite_positive.size == 0:
        return 1.0
    return max(float(np.percentile(finite_positive, 98)), min_visible * 10.0)


def _grid_values(value_range: Tuple[float, float], resolution: float) -> np.ndarray:
    """生成等间距的网格采样点
    
    Args:
        value_range: 采样范围 (start, end)
        resolution: 分辨率（米/点）
        
    Returns:
        一维数组，包含所有采样点的坐标值
    """
    start, end = map(float, value_range)
    if end < start:
        start, end = end, start
    count = int(math.floor((end - start) / resolution)) + 1
    return start + np.arange(count, dtype=float) * resolution


def _default_vehicle(env: Any) -> Optional[Any]:
    """获取默认的主车对象
    
    Args:
        env: MetaDrive环境
        
    Returns:
        主车对象，如果找不到则返回None
    """
    agents = getattr(env, "agents", None)
    if agents:
        return next(iter(agents.values()))
    return getattr(env, "vehicle", None)


def _unit_heading(vehicle: Any) -> np.ndarray:
    """获取车辆的单位朝向向量
    
    Args:
        vehicle: 车辆对象
        
    Returns:
        单位向量 [cos(theta), sin(theta)]
    """
    heading = getattr(vehicle, "heading", None)
    if heading is not None:
        heading = RiskFieldCalculator._xy(heading)
        norm = float(np.linalg.norm(heading))
        if norm > RiskFieldCalculator.EPS:
            return heading / norm
    theta = RiskFieldCalculator._safe_float(getattr(vehicle, "heading_theta", 0.0))
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def _draw_lanes(
    ax: Any,
    env: Any,
    transform: Any,
    plot_bounds: Tuple[float, float, float, float],
    step: float,
    *,
    draw_centers: bool,
    draw_surfaces: bool,
):
    """绘制车道线、车道表面和车道中心线
    
    根据MetaDrive车道的line_types属性区分不同类型的车道线：
    - broken: 虚线（浅灰色，允许变道）
    - solid: 实线（浅红色，禁止变道）
    - boundary: 边界线（红色加粗）
    
    Args:
        ax: matplotlib axes对象
        env: MetaDrive环境
        transform: 坐标转换函数
        plot_bounds: 绘图边界
        step: 车道线采样步长（米）
        draw_centers: 是否绘制车道中心线（仅用于几何调试）
        draw_surfaces: 是否填充车道表面（默认关闭，避免整片路网被填满）
    """
    lanes = _iter_lanes(env)
    surface_label = "lane surface"
    center_label = "lane center"
    edge_label = "lane edge"
    
    for lane in lanes:
        # 采样车道中心、左边缘、右边缘的点
        center, left_edge, right_edge = _sample_lane_points(lane, step)
        if center.size == 0:
            continue

        # 转换到绘图坐标系
        center_plot = _transform_points(center, transform)
        left_plot = _transform_points(left_edge, transform)
        right_plot = _transform_points(right_edge, transform)
        
        # 过滤掉远离视野的车道
        if not _points_in_bounds(center_plot, plot_bounds, margin=8.0):
            continue

        # 绘制车道表面（默认关闭，只在需要检查路面多边形时打开）
        if draw_surfaces:
            surface = np.vstack([left_plot, right_plot[::-1]])
            ax.fill(
                surface[:, 0],
                surface[:, 1],
                color="#94a3b8",              # 浅灰色
                alpha=0.10,                    # 很低透明度
                linewidth=0.0,
                label=surface_label,
                clip_on=True,
                zorder=2,                      # 图层顺序：在热力图之上
            )
            surface_label = "_nolegend_"       # 后续不再添加到图例

        # 绘制车道中心线（青色虚线，仅用于调试）
        if draw_centers:
            ax.plot(
                center_plot[:, 0],
                center_plot[:, 1],
                color="#6ee7b7",              # 青绿色
                linewidth=0.65,
                alpha=0.65,
                linestyle="--",
                label=center_label,
                clip_on=True,
                zorder=3,
            )
            center_label = "_nolegend_"

        # 绘制车道边缘线（根据line_types区分虚线/实线）
        for edge_points, side in ((left_plot, 0), (right_plot, 1)):
            style = _lane_line_style(lane, side)
            if style is None:
                continue
            ax.plot(
                edge_points[:, 0],
                edge_points[:, 1],
                color=style["color"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                linestyle=style["linestyle"],
                label=edge_label,
                clip_on=True,
                zorder=3,
            )
            edge_label = "_nolegend_"


def _iter_lanes(env: Any):
    """迭代环境中所有车道
    
    Args:
        env: MetaDrive环境
        
    Returns:
        车道列表
    """
    road_network = getattr(getattr(env, "current_map", None), "road_network", None)
    if road_network is None:
        return []
    try:
        return list(road_network.get_all_lanes())
    except Exception:
        return []


def _sample_lane_points(lane: Any, step: float):
    """沿车道长度方向采样中心点和边缘点
    
    Args:
        lane: 车道对象
        step: 采样步长（米）
        
    Returns:
        (center_points, left_points, right_points) 三个点数组
    """
    length = RiskFieldCalculator._safe_float(getattr(lane, "length", math.nan))
    if not math.isfinite(length) or length <= 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))

    # 计算采样点数量
    sample_count = max(2, int(math.ceil(length / max(step, 0.1))) + 1)
    longitudinal_values = np.linspace(0.0, length, sample_count)
    center_points = []
    left_points = []
    right_points = []

    # 逐点采样
    for longitudinal in longitudinal_values:
        width = _lane_width(lane, longitudinal)
        if not math.isfinite(width):
            continue
        center_points.append(_lane_position(lane, longitudinal, 0.0))           # 中心
        left_points.append(_lane_position(lane, longitudinal, width / 2.0))     # 左边缘
        right_points.append(_lane_position(lane, longitudinal, -width / 2.0))   # 右边缘

    return np.asarray(center_points), np.asarray(left_points), np.asarray(right_points)


def _lane_width(lane: Any, longitudinal: float) -> float:
    """获取指定纵向位置的车道宽度
    
    Args:
        lane: 车道对象
        longitudinal: 纵向坐标（米）
        
    Returns:
        车道宽度（米）
    """
    try:
        return RiskFieldCalculator._safe_float(lane.width_at(longitudinal))
    except Exception:
        return RiskFieldCalculator._safe_float(getattr(lane, "width", math.nan))


def _lane_position(lane: Any, longitudinal: float, lateral: float) -> np.ndarray:
    """获取车道上指定Frenet坐标的世界坐标
    
    Args:
        lane: 车道对象
        longitudinal: 纵向坐标（米）
        lateral: 横向坐标（米，相对于车道中心）
        
    Returns:
        世界坐标 [x, y]
    """
    try:
        return RiskFieldCalculator._xy(lane.position(longitudinal, lateral))
    except Exception:
        return np.array([math.nan, math.nan], dtype=float)


def _lane_line_style(lane: Any, side: int) -> Optional[Dict[str, Any]]:
    """根据车道线类型返回绘制样式
    
    Args:
        lane: 车道对象
        side: 0表示左侧，1表示右侧
        
    Returns:
        样式字典，包含color、linewidth、alpha、linestyle；如果不应绘制则返回None
    """
    line_types = getattr(lane, "line_types", None)
    line_type = None
    if line_types is not None:
        try:
            line_type = line_types[side]
        except Exception:
            line_type = None

    line_text = str(line_type).lower()
    
    # 未知或无类型的线不绘制
    if "unknown" in line_text or "none" in line_text:
        return None
    
    # 虚线：浅灰短划线，允许变道
    if "broken" in line_text:
        return {
            "color": "#cbd5e1",              # 浅灰色
            "linewidth": 0.65,
            "alpha": 0.80,
            "linestyle": (0, (4.0, 4.0)),    # 虚线模式：4像素线+4像素间隔
        }
    
    # 实线/边界线/护栏：浅红实线，与风险热力图色系保持一致
    if "boundary" in line_text or "solid" in line_text or "guardrail" in line_text:
        return {
            "color": "#ef4444",              # 红色边界
            "linewidth": 1.0,                # 只描边，不铺满
            "alpha": 0.75,
            "linestyle": "-",                # 实线
        }
    
    # 默认样式：浅灰实线
    return {
        "color": "#cbd5e1",
        "linewidth": 0.65,
        "alpha": 0.75,
        "linestyle": "-",
    }


def _draw_objects(
    ax: Any,
    env: Any,
    ego: Any,
    calculator: RiskFieldCalculator,
    transform: Any,
    plot_bounds: Tuple[float, float, float, float],
    *,
    annotate: bool,
):
    """绘制主车、周围车辆和静态障碍物
    
    Args:
        ax: matplotlib axes对象
        env: MetaDrive环境
        ego: 主车对象
        calculator: 风险场计算器
        transform: 坐标转换函数
        plot_bounds: 绘图边界
        annotate: 是否标注名称和速度
    """
    # 绘制主车（蓝色矩形）
    _draw_box(ax, ego, transform, plot_bounds, facecolor="#38bdf8", edgecolor="white", label="ego", annotate=annotate)

    # 绘制周围车辆（黄色矩形）
    vehicle_label = "vehicle"
    for obj in calculator._iter_surrounding_vehicles(env, ego):
        _draw_box(
            ax,
            obj,
            transform,
            plot_bounds,
            facecolor="#fef3c7",            # 浅黄色
            edgecolor="#f59e0b",            # 橙色边框
            label=vehicle_label,
            annotate=annotate,
        )
        vehicle_label = "_nolegend_"         # 只添加一次到图例

    # 绘制静态障碍物（粉色矩形）
    object_label = "traffic object"
    for obj in calculator._iter_static_objects(env):
        _draw_box(
            ax,
            obj,
            transform,
            plot_bounds,
            facecolor="#fb7185",            # 粉红色
            edgecolor="#fff1f2",            # 浅粉边框
            label=object_label,
            annotate=annotate,
        )
        object_label = "_nolegend_"


def _draw_box(
    ax: Any,
    obj: Any,
    transform: Any,
    plot_bounds: Tuple[float, float, float, float],
    *,
    facecolor: str,
    edgecolor: str,
    label: str,
    annotate: bool,
):
    """绘制车辆的矩形包围盒
    
    Args:
        ax: matplotlib axes对象
        obj: 车辆或障碍物对象
        transform: 坐标转换函数
        plot_bounds: 绘图边界
        facecolor: 填充颜色
        edgecolor: 边框颜色
        label: 图例标签
        annotate: 是否标注名称和速度
    """
    import matplotlib.patches as patches

    # 获取物体中心位置
    center = RiskFieldCalculator._xy(getattr(obj, "position", (math.nan, math.nan)))
    if not np.isfinite(center).all():
        return

    # 计算四个角的坐标
    heading = _unit_heading(obj)
    left = np.array([-heading[1], heading[0]], dtype=float)
    length = _dimension(obj, "top_down_length", 4.5)
    width = _dimension(obj, "top_down_width", 2.0)
    corners = np.array([
        center + heading * length / 2.0 + left * width / 2.0,   # 前右
        center + heading * length / 2.0 - left * width / 2.0,   # 前左
        center - heading * length / 2.0 - left * width / 2.0,   # 后左
        center - heading * length / 2.0 + left * width / 2.0,   # 后右
    ])
    
    # 转换到绘图坐标系
    corners_plot = _transform_points(corners, transform)
    
    # 过滤掉远离视野的物体
    if not _points_in_bounds(corners_plot, plot_bounds, margin=3.0):
        return

    # 绘制矩形
    polygon = patches.Polygon(
        corners_plot,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=0.9,
        label=label,
        clip_on=True,
        zorder=5,                          # 最高图层：在所有元素之上
    )
    ax.add_patch(polygon)

    # 添加标注文本（名称 + 速度）
    if annotate:
        label_text = getattr(obj, "name", type(obj).__name__)
        speed = RiskFieldCalculator._safe_float(getattr(obj, "speed", math.nan))
        if math.isfinite(speed):
            label_text = f"{label_text}\n{speed:.1f} m/s"
        text_pos = transform(center)
        ax.text(text_pos[0], text_pos[1], label_text, fontsize=7, color="white", ha="center", va="center")


def _draw_current_risk_box(ax: Any, env: Any, vehicle: Any, calculator: RiskFieldCalculator):
    """在左上角绘制当前风险值的统计框
    
    Args:
        ax: matplotlib axes对象
        env: MetaDrive环境
        vehicle: 主车对象
        calculator: 风险场计算器
    """
    _, info = calculator.calculate(env, vehicle)
    
    # 构建统计文本
    summary = "\n".join([
        f"risk: {info.get('risk_field_cost', 0.0):.3f}",
        f"road: {info.get('risk_field_road_cost', 0.0):.3f}",
        f"offroad: {info.get('risk_field_offroad_cost', 0.0):.2f}",
        f"vehicle: {info.get('risk_field_vehicle_cost', 0.0):.3f}",
        f"object: {info.get('risk_field_object_cost', 0.0):.3f}",
        f"headway: {info.get('risk_field_headway_cost', 0.0):.3f}",
        f"ttc: {info.get('risk_field_ttc_cost', 0.0):.3f}",
        f"on_road: {int(bool(info.get('risk_field_on_road', True)))}",
        f"n_vehicle: {info.get('risk_field_surrounding_vehicle_count', 0)}",
        f"n_object: {info.get('risk_field_surrounding_object_count', 0)}",
    ])
    
    # 在左上角添加文本框
    ax.text(
        0.015,                                    # x位置（相对坐标）
        0.985,                                    # y位置（相对坐标）
        summary,
        transform=ax.transAxes,                   # 使用相对坐标系
        va="top",                                 # 垂直对齐：顶部
        ha="left",                                # 水平对齐：左侧
        fontsize=9,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.35",        # 圆角矩形
            "facecolor": "black",                 # 黑色背景
            "alpha": 0.65,                        # 半透明
            "edgecolor": "white",                 # 白色边框
        },
    )


def _dimension(obj: Any, attr: str, default: float) -> float:
    """安全地获取物体的尺寸属性
    
    Args:
        obj: 物体对象
        attr: 属性名（如"top_down_length"）
        default: 默认值
        
    Returns:
        尺寸值，如果无效则返回默认值
    """
    value = getattr(obj, attr, default)
    if callable(value):
        value = value()
    value = RiskFieldCalculator._safe_float(value)
    return value if math.isfinite(value) and value > RiskFieldCalculator.EPS else default


def _transform_points(points: np.ndarray, transform: Any) -> np.ndarray:
    """批量转换点的坐标
    
    Args:
        points: 点数组 (N, 2)
        transform: 坐标转换函数
        
    Returns:
        转换后的点数组 (N, 2)
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 2))
    return np.asarray([transform(point) for point in points], dtype=float)


def _points_in_bounds(points: np.ndarray, bounds: Tuple[float, float, float, float], margin: float = 0.0) -> bool:
    """检查是否有至少一个点在指定边界内
    
    Args:
        points: 点数组 (N, 2)
        bounds: 边界 (x_min, x_max, y_min, y_max)
        margin: 边界容差（米）
        
    Returns:
        如果有至少一个点在边界内（含容差）则返回True
    """
    if points.size == 0:
        return False
    x_min, x_max, y_min, y_max = bounds
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return False
    points = points[finite]
    return bool(
        ((x_min - margin <= points[:, 0]) & (points[:, 0] <= x_max + margin) &
         (y_min - margin <= points[:, 1]) & (points[:, 1] <= y_max + margin)).any()
    )
