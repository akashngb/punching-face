"""Fuse visible hair masks and unoriented pixel flow into surface-following locks."""
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def orientation_field(rgb):
    """Multiscale structure tensor: ridge direction, confidence, not brightness."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    xx, yy, xy = [sum(cv2.GaussianBlur(v, (0, 0), s) * w
                        for s, w in [(1.2, .5), (2.5, .35), (5, .15)])
                  for v in (gx*gx, gy*gy, gx*gy)]
    theta = .5*np.arctan2(2*xy, xx-yy) + np.pi/2
    confidence = np.sqrt((xx-yy)**2 + 4*xy**2) / np.maximum(xx+yy, 1e-8)
    # Compression/flat patches must not count as a strongly measured direction.
    confidence *= np.clip(np.sqrt(xx+yy)/.12, 0, 1)
    return np.stack([np.cos(theta), np.sin(theta)], -1), confidence


def path_samples(paths, crop):
    points, directions = [], []
    origin, size = np.array(crop[:2]), np.array(crop[2:])-crop[:2]
    for path in paths:
        path = np.asarray(path)*size+origin
        for a, b in zip(path[:-1], path[1:]):
            if np.linalg.norm(b-a) < 1:
                continue
            samples = a + np.linspace(0, 1, max(2, int(np.linalg.norm(b-a)/3)))[:, None]*(b-a)
            points.extend(samples); directions.extend(np.tile(unit(b-a), (len(samples), 1)))
    return (cKDTree(points), np.asarray(directions)) if points else (None, None)


def lift_direction(cam, pose, xy, direction, normal, B):
    """Intersect the image differential with the actual surface tangent plane."""
    ray = np.c_[cam.cam_from_img(xy), np.ones(len(xy))] @ pose.rotation.matrix() @ B.T
    dr = np.c_[cam.cam_from_img(xy+direction)-cam.cam_from_img(xy), np.zeros(len(xy))] @ pose.rotation.matrix() @ B.T
    denom = np.sum(ray*normal, axis=1)
    denom = np.where(abs(denom) < 1e-5, 1e-5, denom)
    return unit(dr-ray*(np.sum(dr*normal, axis=1)/denom)[:, None])


def photo_field(folder, points, normals, spec, rec, center, B, transform, surface=None,colors_only=False):
    world = points/transform['scale'] @ B + center
    n = len(points)
    tensor = np.zeros((n, 3, 3)); color_sum = np.zeros((n, 3))
    support = np.zeros(n); observed = np.zeros(n); votes = np.zeros(n); foreground = np.zeros(n)
    directed = np.zeros((n, 3)); direction_weight = np.zeros(n)
    audit = []
    images = {im.name: im for im in rec.images.values()}
    from scripts.photo_detail import detail_image
    color_best=np.zeros(n);best_colors=np.tile(spec['hair']['colorSrgb'],(n,1))
    for view in spec.get('views', []):
        im = images.get(view['filename'])
        if im is None or not view.get('hairRegions'):
            continue
        px = np.asarray(Image.open(folder/'images'/im.name).convert('RGBA'))
        h, w = px.shape[:2]; crop = np.array(spec['crops'][im.name])
        mask = np.zeros((h, w), np.uint8)
        for poly in view['hairRegions']:
            if len(poly) >= 3:
                cv2.fillPoly(mask, [np.rint(np.asarray(poly)*(crop[2:]-crop[:2])+crop[:2]).astype(np.int32)], 255)
        mask = (mask > 0) & (px[:, :, 3] > 220)
        pose = im.cam_from_world(); cam = rec.cameras[im.camera_id]
        cp = world @ pose.rotation.matrix().T + pose.translation
        xy = cam.img_from_cam(cp)
        ij = np.rint(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(int)
        inside = (ij[:, 0] >= 1) & (ij[:, 0] < w-1) & (ij[:, 1] >= 1) & (ij[:, 1] < h-1) & (cp[:, 2] > 0)
        x = np.clip(ij[:, 0], 0, w-1); y = np.clip(ij[:, 1], 0, h-1)
        facing = np.maximum(np.sum((normals @ B)*unit(im.projection_center()-world), axis=1), 0)
        visibility = inside * facing**3 * (facing > .18)
        if surface is not None:
            from scripts.photo_geometry import zbuffer
            vertices, faces = surface
            vc = (vertices/transform['scale'] @ B+center) @ pose.rotation.matrix().T+pose.translation
            depth = zbuffer(cam.img_from_cam(vc), vc[:, 2], faces, w, h)
            visibility *= np.abs(cp[:, 2]-depth[y, x])*transform['scale'] < .007
        observed += visibility
        foreground += visibility*(px[y, x, 3]>220)
        hit = mask[y, x]
        votes += visibility*hit
        detailed=detail_image(folder,im.name);factor=detailed.shape[1]/w
        dx=np.clip(np.rint(xy[:,0]*factor).astype(int),0,detailed.shape[1]-1)
        dy=np.clip(np.rint(xy[:,1]*factor).astype(int),0,detailed.shape[0]-1)
        color_quality=visibility*hit
        take=color_quality>color_best;best_colors[take]=detailed[dy[take],dx[take],:3]/255.;color_best=np.maximum(color_best,color_quality)
        if colors_only:continue
        flow, confidence = orientation_field(detailed[:, :, :3])
        direction = flow[dy, dx].copy(); confidence = confidence[dy, dx]
        tree, paths = path_samples(view.get('hairFlowPaths', []), crop)
        semantic_weight = np.zeros(n)
        if tree is not None:
            distance, nearest = tree.query(np.nan_to_num(xy, nan=-1e5, posinf=-1e5, neginf=-1e5))
            semantic = paths[nearest]
            direction *= np.where(np.sum(direction*semantic, axis=1) < 0, -1, 1)[:, None]
            semantic_weight = np.exp(-(distance/max(12, (crop[2]-crop[0])*.09))**2)
            blend = semantic_weight*(1-confidence*.75)
            direction = unit(direction*(1-blend[:, None])+semantic*blend[:, None])
        weight = visibility*hit*np.maximum(confidence, semantic_weight*.45)
        valid = weight > .002
        d3 = np.zeros((n, 3))
        if valid.any():
            d3[valid] = lift_direction(cam, pose, xy[valid], direction[valid], normals[valid], B)
        tensor += weight[:, None, None]*d3[:, :, None]*d3[:, None, :]
        directed += d3*(weight*semantic_weight)[:, None]
        direction_weight += weight*semantic_weight
        support += weight
        color_sum += px[y, x, :3]/255 * (visibility*hit)[:, None]
        audit.append({'filename': im.name, 'supportedSamples': int(valid.sum()),
                      'flowPaths': len(view.get('hairFlowPaths', []))})
    values, vectors = np.linalg.eigh(tensor)
    directions = vectors[:, :, -1]
    directions *= np.where(np.sum(directions*directed, axis=1) < 0, -1, 1)[:, None]
    # Votes are weighted by facing, so a grazing/inaccurate mask cannot erase
    # a lock supported by a clear frontal or profile view.
    coverage = np.divide(votes, observed, out=np.zeros(n), where=observed > 1e-8)
    colors = best_colors
    coherence = (values[:, -1]-values[:, -2])/np.maximum(values[:, -1], 1e-8)
    return {'directions': directions, 'colors': colors, 'confidence': np.clip(support, 0, 1)*coherence,
            'coverage': coverage, 'observed': observed, 'foreground': foreground, 'directed': direction_weight > .02, 'views': audit}


def trace_locks(roots, normals, directions, confidence, parameters, hairline, segments=18):
    """Trace both ways through a surface field, stopping at its hair boundary.

    Moving least squares supplies the local tangent plane, avoiding the old
    fixed spherical bend that intersected the cap. Output is a rest-space
    offset curve, bound to the original triangle like the legacy strand groom.
    """
    tree = cKDTree(roots)
    offsets = np.zeros((len(roots), segments+1, 3), np.float32)
    surface_normals = np.zeros_like(offsets)
    top = np.clip((roots[:, 1]-hairline+.025)/.055, 0, 1)
    posterior = np.clip((-normals[:, 2]-.15)/.65, 0, 1)*(1-np.clip(normals[:, 1], 0, 1))
    top *= 1-posterior
    full_length = (parameters['sideLengthMm']*(1-top)+parameters['lengthMm']*top)*.001
    # Exterior visible sections of longer strands, not invented scalp roots.
    fraction=parameters.get('visibleSectionFraction',.55)
    lengths = np.clip(full_length*fraction, .004, .055)
    step = lengths/segments
    mid = segments//2
    surface_normals[:, mid] = normals
    for sign in (-1, 1):
        q = roots.copy(); previous = directions*sign
        for j in range(1, mid+1):
            proposed = q+previous*step[:, None]
            distances, near = tree.query(proposed, k=min(12, len(roots)))
            if distances.ndim == 1:
                distances=distances[:, None]; near=near[:, None]
            weights = np.exp(-((distances/np.maximum(distances[:, -1:], .002))**2)*3)
            weights /= weights.sum(1, keepdims=True)
            n = unit(np.sum(normals[near]*weights[:, :, None], axis=1))
            center = np.sum(roots[near]*weights[:, :, None], axis=1)
            projected = proposed-n*np.sum((proposed-center)*n, axis=1)[:, None]
            # Stop before a lock leaves the observed region onto forehead/ear.
            valid = distances[:, 0] < .0035
            q = np.where(valid[:, None], projected, q)
            local = directions[near].copy()
            local *= np.where(np.sum(local*previous[:, None], axis=2) < 0, -1, 1)[:, :, None]
            flow = unit(np.sum(local*weights[:, :, None], axis=1))
            previous = unit(previous*.65+flow*.35)
            previous = unit(previous-n*np.sum(previous*n, axis=1)[:, None])
            offsets[:, mid+sign*j] = q-roots
            surface_normals[:, mid+sign*j] = n
    return offsets, surface_normals, lengths


def bind_curve_points(vertices, faces, points):
    """Bind every curve station to the existing deformable surface."""
    import trimesh
    triangles=vertices[faces]
    _,near=cKDTree(triangles.mean(1)).query(points,k=min(8,len(faces)))
    if near.ndim==1:near=near[:,None]
    candidates=triangles[near].reshape(-1,3,3)
    repeated=np.repeat(points,near.shape[1],axis=0)
    closest=trimesh.triangles.closest_point(candidates,repeated)
    distance=np.linalg.norm(closest-repeated,axis=1).reshape(near.shape)
    choice=np.argmin(distance,axis=1);triangle=near[np.arange(len(points)),choice]
    closest=closest.reshape(len(points),near.shape[1],3)[np.arange(len(points)),choice]
    bary=trimesh.triangles.points_to_barycentric(triangles[triangle],closest)
    bary=np.clip(bary,0,1);bary/=bary.sum(1,keepdims=True)
    return {'triangles':faces[triangle].ravel().tolist(),'weights':np.round(bary,7).ravel().tolist()}
