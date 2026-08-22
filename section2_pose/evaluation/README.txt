Run from repository root:
PYTHONPATH=. python section2_pose/evaluation/calibration_synthetic.py
PYTHONPATH=. python section2_pose/evaluation/synthetic_pose_eval.py --n 100 --envelope-trials 100
PYTHONPATH=. python section2_pose/evaluation/uncertainty_mc.py
PYTHONPATH=. python section2_pose/evaluation/plot_results.py

The synthetic/reference camera is 640x480 with fx=fy=450, cx=320, cy=240.
These are controlled-evaluation assumptions, not physical deployment calibration.
