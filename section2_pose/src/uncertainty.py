from __future__ import annotations
import numpy as np
from .pose_geometry import CameraModel, PalletModel, Pose2D, estimate_pose


def monte_carlo_uncertainty(observed_px, camera, pallet, initial, *, samples=500,
                            pixel_sigma_px=1.0, height_sigma_m=0.01, tilt_sigma_deg=0.5,
                            seed=42):
    rng=np.random.default_rng(seed); vals=[]
    observed_px=np.asarray(observed_px,float).reshape(2,2)
    for _ in range(samples):
        cam=CameraModel(camera.fx,camera.fy,camera.cx,camera.cy,
                        camera.height_m+rng.normal(0,height_sigma_m),
                        camera.tilt_deg+rng.normal(0,tilt_sigma_deg))
        obs=observed_px+rng.normal(0,pixel_sigma_px,(2,2))
        est,_,_=estimate_pose(obs,cam,pallet,initial=initial)
        vals.append([est.x_m,est.y_m,est.theta_deg])
    a=np.asarray(vals)
    q=np.percentile(a,[2.5,50,97.5],axis=0)
    return {
        'samples': int(samples),
        'std': {'x_m':float(a[:,0].std()),'y_m':float(a[:,1].std()),'theta_deg':float(a[:,2].std())},
        'p95_interval': {
            'x_m':[float(q[0,0]),float(q[2,0])],
            'y_m':[float(q[0,1]),float(q[2,1])],
            'theta_deg':[float(q[0,2]),float(q[2,2])]
        },
        'p95_half_width': {
            'x_m':float((q[2,0]-q[0,0])/2),
            'y_m':float((q[2,1]-q[0,1])/2),
            'theta_deg':float((q[2,2]-q[0,2])/2)
        }
    }
