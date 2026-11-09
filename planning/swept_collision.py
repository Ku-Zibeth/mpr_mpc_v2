"""Conservative continuous OBB contact along piecewise-linear pose intervals.

Translation can follow the kinematic circular arc, including linear speed
change, or a supplied linear pose interval. Rotating corners are enclosed by a disk inflation around the
mid-heading OBB. Only ambiguous rotating intervals are recursively bisected;
unresolved leaves remain hard collisions, never optimistic safe results.
"""

import math

import torch


def _interval_sat(r0, r1, yaw0, yaw1, other_yaw, ego_length, ego_width, lengths, widths, position_bound):
    middle = (yaw0 + yaw1) / 2
    forward = torch.stack((middle.cos(), middle.sin()), -1)
    normal = torch.stack((-middle.sin(), middle.cos()), -1)
    of = torch.stack((other_yaw.cos(), other_yaw.sin()), -1)
    on = torch.stack((-other_yaw.sin(), other_yaw.cos()), -1)
    axes = torch.stack((forward, normal, of, on), 1)
    radius = (ego_length/2 * (forward[:, None] * axes).sum(-1).abs()
              + ego_width/2 * (normal[:, None] * axes).sum(-1).abs()
              + lengths[:, None]/2 * (of[:, None] * axes).sum(-1).abs()
              + widths[:, None]/2 * (on[:, None] * axes).sum(-1).abs())
    rotation_bound = math.hypot(ego_length, ego_width) * ((yaw1-yaw0).abs().clamp_max(2*math.pi)/4).sin()
    radius = radius + rotation_bound[:, None] + position_bound[:, None] + 1e-6
    distance = (r0[:, None] * axes).sum(-1)
    change = ((r1-r0)[:, None] * axes).sum(-1)
    moving = change.abs() > 1e-12
    divisor = torch.where(moving, change, torch.ones_like(change))
    first, last = (-radius-distance)/divisor, (radius-distance)/divisor
    entry = torch.where(moving, torch.minimum(first, last),
                        torch.where(distance.abs() <= radius, -torch.inf, torch.inf)).amax(-1).clamp_min(0)
    exit = torch.where(moving, torch.maximum(first, last),
                       torch.where(distance.abs() <= radius, torch.inf, -torch.inf)).amin(-1).clamp_max(1)
    return entry <= exit, entry


def _pose(u, other_start, other_delta, ego_delta, yaw_start, yaw_delta, speed0, speed1, curved):
    speed_sum = speed0+speed1
    progress = torch.where(speed_sum > 1e-12,
        u*(2*speed0+(speed1-speed0)*u)/speed_sum.clamp_min(1e-12), u).clamp(0, 1)
    arc = curved & (yaw_delta.abs() > 1e-6) & (yaw_delta.abs() < math.pi)
    denominator = torch.where(arc, (yaw_delta/2).sin(), torch.ones_like(yaw_delta))
    along = ((progress-.5)*yaw_delta).sin()/(2*denominator)+.5
    across = -(progress*yaw_delta/2).sin()*((1-progress)*yaw_delta/2).sin()/denominator
    normal_delta = torch.stack((-ego_delta[:, 1], ego_delta[:, 0]), -1)
    center = ego_delta * torch.where(arc, along, progress)[:, None]
    center += normal_delta * torch.where(arc, across, 0.)[:, None]
    return other_start + u[:, None]*other_delta - center, yaw_start+progress*yaw_delta


def _position_bound(chord, span, u0, u1, speed0, speed1, curved):
    angle = span.abs()
    arc = curved & (angle < math.pi)
    sagitta = torch.where(arc, chord*(angle/4).tan()/2, 0.)
    denominator = 2*(angle/2).sin()
    arc_length = torch.where(arc & (angle > 1e-6), chord*angle/denominator.clamp_min(1e-12), chord)
    v0, v1 = speed0+(speed1-speed0)*u0, speed0+(speed1-speed0)*u1
    acceleration_bound = arc_length * (v1-v0).abs()/(4*(v0+v1).clamp_min(1e-12))
    return sagitta+acceleration_bound


def swept_collisions(positions, headings, other_xy, other_v, other_heading, lengths, widths,
                     *, ego_length, ego_width, dt, max_depth=10, angle_tolerance=1e-4,
                     position_tolerance=1e-4, skip=None, curved=False, speeds=None,
                     obstacle_acceleration_bound=0., obstacle_velocity_error=0.):
    if (not math.isfinite(obstacle_acceleration_bound) or obstacle_acceleration_bound < 0
            or not math.isfinite(obstacle_velocity_error) or obstacle_velocity_error < 0):
        raise ValueError("Obstacle uncertainty bounds must be finite and nonnegative.")
    tensors = (positions, headings, other_xy, other_v, other_heading, lengths, widths)
    dtype = torch.promote_types(positions.dtype, torch.float32)
    for tensor in (*tensors, *((speeds,) if speeds is not None else ())):
        dtype = torch.promote_types(dtype, tensor.dtype)
    positions, headings, other_xy, other_v, other_heading, lengths, widths = (
        tensor.to(device=positions.device, dtype=dtype) for tensor in tensors)
    if speeds is not None:
        speeds = speeds.to(device=positions.device, dtype=dtype)
    n, samples, _ = positions.shape
    intervals = samples-1
    times = positions.new_tensor(range(intervals)) * dt
    r0 = other_xy[None, None] + times[None, :, None, None] * other_v[None, None] - positions[:, :-1, None]
    r1 = other_xy[None, None] + (times+dt)[None, :, None, None] * other_v[None, None] - positions[:, 1:, None]
    change = r1-r0
    ego_delta = positions[:, 1:]-positions[:, :-1]
    yaw_delta = headings[:, 1:]-headings[:, :-1]
    speed0 = speeds[:, :-1] if speeds is not None else torch.ones_like(yaw_delta)
    speed1 = speeds[:, 1:] if speeds is not None else torch.ones_like(yaw_delta)
    curve_mask = (yaw_delta.abs() < math.pi) & bool(curved)
    position_bound = _position_bound(ego_delta.norm(dim=-1), yaw_delta, torch.zeros_like(yaw_delta),
        torch.ones_like(yaw_delta), speed0, speed1, curve_mask)
    fraction = (-(r0*change).sum(-1)/change.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
    closest = r0 + fraction[..., None] * change
    # A bounded acceleration/velocity error adds an isotropic position tube
    # around CV. Use the interval-end radius throughout the swept interval,
    # including recursive leaves: never interpolate a smaller unsafe bound.
    uncertainty = obstacle_velocity_error*(times+dt)+.5*obstacle_acceleration_bound*(times+dt).square()
    inflated_lengths = lengths[None]+2*uncertainty[:, None]
    inflated_widths = widths[None]+2*uncertainty[:, None]
    circles = math.hypot(ego_length, ego_width)/2 + torch.sqrt(inflated_lengths.square()+inflated_widths.square())/2
    possible = closest.square().sum(-1) <= (circles[None]+position_bound[..., None]+1e-6).square()
    if skip is not None:
        possible &= ~skip[:, None, None]
    indices = possible.nonzero()
    first_contact = positions.new_full((n*intervals,), torch.inf)
    if not len(indices):
        return first_contact.view(n, intervals)
    ni, ti, oi = indices.unbind(1)
    origin = ni*intervals + ti
    a, b = r0[ni, ti, oi], r1[ni, ti, oi]
    y0, y1 = headings[ni, ti], headings[ni, ti+1]
    root_other, root_other_delta = a.clone(), other_v[oi]*dt
    root_ego, root_yaw, root_turn = ego_delta[ni, ti], y0.clone(), yaw_delta[ni, ti]
    root_v0, root_v1, root_curve = speed0[ni, ti], speed1[ni, ti], curve_mask[ni, ti]
    o, length, width = other_heading[oi], inflated_lengths[ti, oi], inflated_widths[ti, oi]
    u0, u1 = torch.zeros_like(y0), torch.ones_like(y0)
    for depth in range(max_depth+1):
        chord = (root_other_delta*(u1-u0)[:, None]-(b-a)).norm(dim=-1)
        pad = _position_bound(chord, y1-y0, u0, u1, root_v0, root_v1, root_curve)
        hit, entry = _interval_sat(a, b, y0, y1, o, ego_length, ego_width, length, width, pad)
        leaf = hit & ((((y1-y0).abs() <= angle_tolerance) & (pad <= position_tolerance)) | (depth == max_depth))
        first_contact.scatter_reduce_(0, origin[leaf], (u0+entry*(u1-u0))[leaf], reduce="amin", include_self=True)
        split = (hit & ~leaf).nonzero().flatten()
        if not len(split):
            break
        a, b, y0, y1, o, length, width, origin, u0, u1 = (
            v[split] for v in (a, b, y0, y1, o, length, width, origin, u0, u1))
        root_other, root_other_delta, root_ego, root_yaw, root_turn, root_v0, root_v1, root_curve = (
            v[split] for v in (root_other, root_other_delta, root_ego, root_yaw, root_turn, root_v0, root_v1, root_curve))
        um = (u0+u1)/2
        mid, ym = _pose(um, root_other, root_other_delta, root_ego, root_yaw, root_turn, root_v0, root_v1, root_curve)
        a, b = torch.cat((a, mid)), torch.cat((mid, b))
        y0, y1 = torch.cat((y0, ym)), torch.cat((ym, y1))
        u0, u1 = torch.cat((u0, um)), torch.cat((um, u1))
        o, length, width, origin = (v.repeat(2) for v in (o, length, width, origin))
        root_other, root_other_delta, root_ego = (v.repeat(2, 1) for v in (root_other, root_other_delta, root_ego))
        root_yaw, root_turn, root_v0, root_v1, root_curve = (v.repeat(2) for v in (root_yaw, root_turn, root_v0, root_v1, root_curve))
    return first_contact.view(n, intervals)
