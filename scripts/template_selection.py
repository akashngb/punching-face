"""Choose identity-fit strength using only an inner split of geometry frames."""

import numpy as np
from scripts.fit_head_template import fit_template
from scripts.photo_geometry import camera_rays, robust_landmarks
from scripts.surface_evidence import surface_projection_error

CANDIDATES = (0.025, 0.01, 0.005, 0.001)


def normalized_frame(points):
    center = (points[10] + points[152]) / 2
    right = points[263] - points[33]
    right = right / np.linalg.norm(right)
    up = points[10] - points[152]
    up -= right * np.dot(up, right)
    up /= np.linalg.norm(up)
    basis = np.stack([right, up, np.cross(right, up)])
    if (points[1] - (points[33] + points[263]) / 2) @ basis[2] < 0:
        raise ValueError('The recovered face orientation is inconsistent.')
    aligned = (points - center) @ basis.T
    scale = 0.20 / (aligned[10, 1] - aligned[152, 1])
    return aligned * scale, center, basis, {'scale': float(scale)}


def choose_regularization(rec, frames, training):
    """Never use the outer withheld views to select fitting parameters."""
    ordered = sorted(training, key=lambda im: frames[im.name].get('yaw') or 0)
    if len(ordered) < 15:
        return 0.025, {
            'available': False,
            'reason': 'Too few training views for a separate parameter-selection split.',
        }
    validation = ordered[1::5]
    names = {im.name for im in validation}
    fit = [im for im in ordered if im.name not in names]
    report = {
        'available': True,
        'trainingFrames': [im.name for im in fit],
        'selectionFrames': [im.name for im in validation],
        'outerWithheldUsed': False,
        'candidates': [],
    }
    try:
        points, _ = robust_landmarks(*camera_rays(rec, frames, fit))
        normalized, center, basis, transform = normalized_frame(points)
    except ValueError as error:
        report.update(available=False, reason=str(error))
        return 0.025, report
    for strength in CANDIDATES:
        try:
            vertices, _, _, info = fit_template(normalized, strength)
            score = surface_projection_error(
                vertices,
                info['landmarkBindings'],
                rec,
                frames,
                validation,
                center,
                basis,
                transform,
            )
            report['candidates'].append(
                {
                    'regularization': strength,
                    **score,
                    'score': score['medianPx'] + 0.2 * score['p95Px'],
                }
            )
        except ValueError as error:
            report['candidates'].append(
                {'regularization': strength, 'rejected': str(error)}
            )
    usable = [c for c in report['candidates'] if 'score' in c]
    if not usable:
        return 0.025, {
            **report,
            'available': False,
            'reason': 'No candidate explained the selection frames.',
        }
    baseline = next((c for c in usable if c['regularization'] == 0.025), None)
    ranked = sorted(usable, key=lambda c: c['score'])
    # Require a meaningful validation gain and do not trade a much worse tail
    # for a slightly better median. Close scores retain stronger regularization.
    if baseline:
        ranked = [
            c
            for c in ranked
            if c['score'] < baseline['score'] * 0.98
            and c['p95Px'] <= baseline['p95Px'] * 1.03
        ]
    chosen = ranked[0]['regularization'] if ranked else 0.025
    report['admissibleRegularizations'] = [c['regularization'] for c in ranked] + [
        0.025
    ]
    report['selectedRegularization'] = chosen
    return chosen, report


def fit_selected_template(normalized, rec, frames, training):
    strength, selection = choose_regularization(rec, frames, training)
    for strength in selection.get('admissibleRegularizations', [strength]):
        try:
            result = fit_template(normalized, strength, check_regularization=True)
            selection['selectedRegularization'] = strength
            break
        except ValueError as error:
            if strength == 0.025:
                raise
            selection.setdefault('rejectedBySurfaceQuality', []).append(
                {'regularization': strength, 'reason': str(error)}
            )
    result[3]['selection'] = selection
    return result
