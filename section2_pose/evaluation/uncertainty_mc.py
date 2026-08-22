from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from src.pose_geometry import *


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='artifacts/evaluation/uncertainty.json'); ap.add_argument('--samples',type=int,default=1000); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    rng=np.random.default_rng(args.seed); pallet=PalletModel(); gt=Pose2D(3.0,0.8,30.0)
    base=CameraModel(900,900,640,360,1.2,20.0); obs=project_world(pallet_front_edge_points(gt,pallet),base)
    xs=[]
    for _ in range(args.samples):
        cam=CameraModel(base.fx,base.fy,base.cx,base.cy,base.height_m+rng.normal(0,0.01),base.tilt_deg+rng.normal(0,0.5))
        o=obs+rng.normal(0,1.0,obs.shape)
        est,_,_=estimate_pose(o,cam,pallet,initial=gt); xs.append([est.x_m,est.y_m,est.theta_deg])
    a=np.asarray(xs); result={'ground_truth':{'x_m':gt.x_m,'y_m':gt.y_m,'theta_deg':gt.theta_deg},'samples':args.samples,'std':{'x_m':float(a[:,0].std()),'y_m':float(a[:,1].std()),'theta_deg':float(a[:,2].std())},'p95_interval':{'x_m':[float(np.percentile(a[:,0],2.5)),float(np.percentile(a[:,0],97.5))],'y_m':[float(np.percentile(a[:,1],2.5)),float(np.percentile(a[:,1],97.5))],'theta_deg':[float(np.percentile(a[:,2],2.5)),float(np.percentile(a[:,2],97.5))]}}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
