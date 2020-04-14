
import logging
import numpy as np
import tensorflow as tf
import pdb


log = logging.getLogger(__file__)

def get_2d_general_from_two_points(x0, x1, y0, y1):
    if np.isclose(y0, y1):
        a = 0
        b = 1
        c = -y0
    else:
        a = 1
        b = -(x0-x1)/(y0-y1)
        c = -b*y0 - x0
    return (a, b, c)

def get_2d_general_from_two_points_tf(x0, x1, y0, y1):
    equal = tf.tile(tf.equal(y0, y1)[..., None], (1, 1, 3))
    zeros = tf.zeros_like(x0)
    ones = tf.ones_like(x0)
    b = -(x0-x1)/(y0-y1)
    abc_equal = tf.stack((zeros, ones, -y0), axis=-1)
    abc_neq = tf.stack((ones, b, -b*y0 - x0), axis=-1)
    return tf.where(equal, abc_equal, abc_neq)

# def on_segment(p, q, r):
#     o = tf.ones(shape=p[..., 0].shape, dtype=tf.bool)
#     z = tf.zeros_like(o)
#     q[...,0] <

def orientation(p, q, r):
    qp = q - p
    rq = r - q
    val = qp[..., 1] * rq[..., 1] - qp[..., 0] * rq[..., 0]
    z = tf.zeros_like(qp[..., 0])
    o = tf.ones_like(z)
    t = 2 * tf.ones_like(z)
    gz = tf.where(val > 0, o, t)
    return tf.where(val > 0, z, gz)

def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, p1)
    intersect = tf.logical_and(tf.not_equal(o1, o2), tf.not_equal(o3, o4))

    ones = tf.ones_like(intersect)
    zeros = tf.zeros_like(intersect) 
    intersect = tf.logical_or(intersect, tf.where(tf.equal(o1, 0)))

def point_intersects_right(segment_start, segment_end, point, prefix='bka'):
    ydiff = tf.abs(segment_end[..., 1] - segment_start[..., 1])
    segment_not_horizontal = ydiff > 1e-8
    u_intersect = (point[..., 1] - segment_start[..., 1]) / (segment_end[..., 1] - segment_start[..., 1])
    xy_intersect = segment_start + tf.einsum('{0},{0}d->{0}d'.format(prefix), u_intersect, (segment_end - segment_start))
    segments_intersect_mask = tf.logical_and(u_intersect >= 0, u_intersect <= 1)
    inside_and_right = tf.logical_and(segments_intersect_mask, xy_intersect[..., 0] > point[..., 0])
    return tf.logical_and(inside_and_right, segment_not_horizontal)

def intersects_q_line(start, end, point):    
    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    # random gradient to avoid parallel lines, 
    r = 0.95857964  # random positive real > 0. https://www.random.org/    

    o1 = r*start[..., 0] + point[..., 1] - start[..., 1] - r*point[..., 0]
    o2 = r*end[..., 0] + point[..., 1] - end[..., 1] - r*point[..., 0]
    o3 = (end[..., 1] - start[..., 1]) * (point[..., 0] - end[..., 0]) - (end[..., 0] - start[..., 0]) * (point[..., 1] - end[1])
    o4 = (end[1] - start[..., 1]) - r*(end[..., 0] - start[..., 0])
    return (o1 * o2 < 0.) and (o3 * o4 < 0.)

def inside_polygon(edges, point, prefix='bka', return_metadata=False):
    starts = edges[..., :-1, :]
    ends = edges[..., 1:, :]
    n_edges = starts.shape[-2].value
    intersects = tf.stack([point_intersects_right(starts[..., i, :], ends[..., i, :], point, prefix=prefix) for i in range(n_edges)], axis=-1)
    intersects_count = tf.reduce_sum(tf.cast(intersects, tf.int32), axis=-1)
    count_mod = tf.math.floormod(intersects_count, 2)
    inside = tf.equal(count_mod, 1)
    if return_metadata:
        return inside, starts, ends
    else:
        return inside

def create_arc(A, B, N=10):
    vec = B - A
    arc_left = vec[1] > 0
    d = np.linalg.norm(vec)
    d2 = d / np.sqrt(2)
    offset = np.asarray([0, d2])
    if arc_left:
        angles = np.linspace(-np.pi/2, 0, N)
        origin = A + offset
    else:
        angles = np.linspace(np.pi/2, 0, N)
        origin = A - offset
    c = np.cos(angles)
    s = np.sin(angles)
    arc = np.around(d2 * np.stack((c, s), axis=-1) + origin, 2)
    # assert(np.allclose(arc[0], A))
    # assert(np.allclose(arc[-1], B))
    return arc

def create_region_from_route(route_2d, d=3, clip_K=10, right_d=None, left_d=None):
    """

    :param route_2d: sequence of 2d points
    :param d: default polygon width
    :param clip_K: max number of points
    :param right_d: optional right width
    :param left_d: optional left width
    :returns: 
    :rtype: 

    """
    assert(route_2d.ndim == 2)
    assert(route_2d.shape[-1] == 2)
    # TODO check if any duplicate points in input!
    
    As = route_2d[:-2]
    Bs = route_2d[1:-1]
    Cs = route_2d[2:]
    A_to_Bs = Bs - As
    B_to_Cs = Cs - Bs
    A_to_Cs = Cs - As

    frames = np.stack((A_to_Bs, B_to_Cs), axis=-1)
    dets = np.linalg.det(frames)
    right_handed_mask = dets > 0
    # TODO ... should this be <= ?
    left_handed_mask = dets < 0
    
    a_lengths = np.linalg.norm(A_to_Bs, axis=-1)
    b_lengths = np.linalg.norm(B_to_Cs, axis=-1)
    c_lengths = np.linalg.norm(A_to_Cs, axis=-1)

    # Fails if there are duplicate sequential points.
    assert(not np.isclose(a_lengths, 0).any())
    assert(not np.isclose(b_lengths, 0).any())
    assert(not np.isclose(c_lengths, 0).any())
    
    a2s = a_lengths ** 2
    b2s = b_lengths ** 2
    c2s = c_lengths ** 2
    numdems = (c2s - a2s - b2s)/(-2*a_lengths*b_lengths)
    thetas = np.arccos(numdems)

    thetas[np.isnan(thetas)] = np.pi
    scales = (np.sqrt(2)-1)*np.sin(thetas) + 1.
    scales = np.tile(scales[:, None], (1, 2))

    thetas_right = thetas.copy()
    thetas_left = thetas.copy()
    thetas_right[right_handed_mask] = 2 * np.pi - thetas_right[right_handed_mask]
    thetas_left[left_handed_mask] = 2 * np.pi - thetas[left_handed_mask]

    theta_2s_right = thetas_right / 2.
    theta_2s_left = thetas_left / 2.

    def R(theta):
        return np.asarray([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    # (P, 2, 2)
    Rs_clock = np.stack([R(-theta) for theta in theta_2s_right], axis=0)
    Rs_counterclock = np.stack([R(theta) for theta in theta_2s_left], axis=0)

    B_to_Cs_norm = np.einsum('pi,p->pi', B_to_Cs, 1./b_lengths)
    if right_d is None:
        right_d = d
    if left_d is None:
        left_d = d
    Right_shift = np.einsum('pij,pj->pi', Rs_clock, B_to_Cs_norm) * right_d
    Left_shift = np.einsum('pij,pj->pi', Rs_counterclock, B_to_Cs_norm) * left_d
    Left_shift *= scales
    Right_shift *= scales
    Right_points = Bs + Right_shift
    Left_points = Bs + Left_shift

    polygon = np.concatenate((Right_points, Left_points[::-1]), axis=0)
    polygon_clip = np.concatenate((Right_points[:clip_K], Left_points[:clip_K][::-1]), axis=0)
    polygon_pp, _ = postprocess_polygon(polygon_clip)
    return polygon_clip, polygon, Right_points, Left_points, A_to_Cs, B_to_Cs, Right_shift, Left_shift

def preprocess_route_2d_for_polygon(route_2d, clip_K=10):
    # ahead = np.where(route_2d[:, 0] >= 0)[0]
    end_diff = route_2d[-1] - route_2d[-2]
    end_vec = end_diff / np.linalg.norm(end_diff)

    end = route_2d[-1] + end_vec
    # zz = np.zeros(shape=(1,2))
    # start_diff = route_2d[1] - zz
    # start_vec = -start_diff / np.linalg.norm(start_diff)
    # start = zz + start_vec

    # preproc = np.concatenate((start, zz, route_2d, end[None]),axis=0)
    preproc = np.concatenate((route_2d, end[None]),axis=0)
    
    if preproc.shape[0] < clip_K:
        print("Adding extra points at the end")
        n_needed = clip_K - preproc.shape[0]
        extras = []
        last = end
        for n in range(n_needed):
            last = end_vec + last.copy()
            extras.append(last)
        preproc = np.concatenate((preproc, np.stack(extras, axis=0)),axis=0)
    return preproc

def postprocess_polygon(polygon):
    linear = np.zeros(shape=(polygon.shape[0]), dtype=np.float32)
    for idx in range(1, polygon.shape[0]-1):
        x0 = polygon[idx-1]
        x1 = polygon[idx]
        x2 = polygon[idx+1]
        v0 = x1 - x0
        v1 = x2 - x1
        v0n = np.linalg.norm(v0)
        v1n = np.linalg.norm(v1)
        v0_norm = v0 / v0n
        v1_norm = v1 / v1n
        if np.isclose(np.dot(v0_norm, v1_norm), 1, atol=1e-3):
            linear[idx] = 1
    return polygon[np.where(linear == 0)], linear

def trim_left_right_polygon(polygon, K):
    if K <= 0:
        log.error("Can't trim 0 or less points!")
        return polygon
    else:
        return polygon[K:-K]

def generate_unsticking_polygon(polygon, min_dist=4.):
    log.info("Generating unsticking polygon.")
    # TODO this assumes the vehicle has some points close to it!
    dists = np.linalg.norm(polygon[:,:2], axis=-1)

    assert(dists.shape[0] % 2 == 0)
    first_half = dists[:dists.shape[0] // 2]
    first_half_close_indicators = first_half < min_dist
    # Searches backwards for index of first closest point in order to get index of one beyond it
    forward_point_idx = first_half_close_indicators.shape[0] - np.argmax(first_half_close_indicators[::-1])
    
    # This will fail if the polygon extends significantly far behind the vehicle.    
    # forward_point_idx = np.argmax(dists > min_dist)
    
    # Trim the points evenly from left and right.
    trimmed = trim_left_right_polygon(polygon, K=forward_point_idx)
    if trimmed.shape[0] < 4:
        # Shouldn't happen, but if it does, the furthest points on the inputppp polygon are probably too close.
        log.error("Can't generate unsticking polygon without creating a degenerate region!")
        return polygon
    return trimmed

def generate_stopping_polygon(d=3):
    log.info("Generating stopping polygon.")
    back_right = np.asarray([-1, d])
    front_right = np.asarray([1, d])
    front_left = np.asarray([1, -d])
    back_left = np.asarray([-1, -d])
    return np.stack((back_right, front_right, front_left, back_left), axis=0)

def create_straight_region_seed(K, d, x_back=-4, x_front=20):
    x = np.linspace(x_back, x_front, K+2)
    y = np.zeros_like(x)
    route_2d = np.stack((x,y), axis=-1)
    return route_2d

def create_left_region_seed(K, d, radius=20, x_back=-4):
    angles = np.linspace(-np.pi/2, np.pi/8, K+1)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    x = np.concatenate(([x_back], x),axis=0)
    y += radius
    y = -1 * np.concatenate(([0], y),axis=0)
    route_2d = np.stack((x, y), axis=-1)
    return route_2d

def create_right_region_seed(K, d, radius=20, x_back=-4):
    angles = np.linspace(np.pi/2, 0, K+1)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    x = np.concatenate(([x_back], x),axis=0)
    y -= radius
    y = -1 * np.concatenate(([0], y),axis=0)
    route_2d = np.stack((x, y), axis=-1)
    return route_2d
