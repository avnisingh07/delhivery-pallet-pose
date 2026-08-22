from __future__ import annotations
from dataclasses import asdict
import math
from .pose_geometry import CameraModel, PalletModel, Pose2D, estimate_pose
from .uncertainty import monte_carlo_uncertainty


def assess_g0g1(g0, g1, camera, pallet, *, pixel_sigma_px=1.0, height_sigma_m=0.01,
                tilt_sigma_deg=0.5, mc_samples=500, max_reprojection_px=3.0,
                initial=None, envelope_max_range_m=2.0):
    """Fixed Section-1 -> Section-2 adapter.

    g0: left endpoint [u,v], g1: right endpoint [u,v].
    """
    if g0 is None or g1 is None:
        return {'pose':None,'status':'INSUFFICIENT_GEOMETRY','reason':'G0_G1_MISSING'}
    try:
        obs=[[float(g0[0]),float(g0[1])],[float(g1[0]),float(g1[1])]]
        if math.dist(obs[0],obs[1]) < 5.0:
            return {'pose':None,'status':'INSUFFICIENT_GEOMETRY','reason':'FRONT_EDGE_TOO_SHORT'}
        est,rms,_=estimate_pose(obs,camera,pallet,initial=initial)
        unc=monte_carlo_uncertainty(obs,camera,pallet,est,samples=mc_samples,pixel_sigma_px=pixel_sigma_px,
                                    height_sigma_m=height_sigma_m,tilt_sigma_deg=tilt_sigma_deg)
        reasons=[]
        if rms > max_reprojection_px: reasons.append('HIGH_REPROJECTION_RESIDUAL')
        if est.x_m > envelope_max_range_m: reasons.append('OUTSIDE_CONSERVATIVE_ENVELOPE')
        if unc['p95_half_width']['x_m'] > 0.02 or unc['p95_half_width']['y_m'] > 0.02: reasons.append('TRANSLATION_UNCERTAINTY_EXCEEDS_2CM')
        if unc['p95_half_width']['theta_deg'] > 3.0: reasons.append('ROTATION_UNCERTAINTY_EXCEEDS_3DEG')
        status='RELIABLE' if not reasons else 'UNRELIABLE'
        return {
            'pose': {'frame':'floor','x_m':est.x_m,'y_m':est.y_m,'theta_deg':est.theta_deg,
                     'face_identity':{'front':'VISIBLE_FRONT','rear':'OPPOSITE_FACE','left':'LEFT','right':'RIGHT'}},
            'status':status,
            'reason':None if status=='RELIABLE' else ';'.join(reasons),
            'quality':{'reprojection_rms_px':rms,'front_edge_length_px':math.dist(obs[0],obs[1])},
            'uncertainty':unc
        }
    except Exception as exc:
        return {'pose':None,'status':'UNRELIABLE','reason':f'POSE_ESTIMATION_ERROR:{type(exc).__name__}'}
