# Integration into the existing repository

Do not modify `section1_detection/`.

Copy these files into the repository:

```text
section2_pose/src/pose_geometry.py
section2_pose/src/uncertainty.py
section2_pose/src/assessment.py
section2_pose/evaluation/synthetic_pose_eval.py
section2_pose/evaluation/calibration_synthetic.py
section2_pose/evaluation/uncertainty_mc.py
section2_pose/evaluation/plot_results.py
section2_pose/configs/pose.yaml
section2_pose/SECTION2_METHOD.md
```

Run from the repository root:

```bash
mkdir -p section2_pose
cp -R /path/to/this/package/src section2_pose/
cp -R /path/to/this/package/evaluation section2_pose/
cp -R /path/to/this/package/configs section2_pose/
cp /path/to/this/package/SECTION2_METHOD.md section2_pose/

PYTHONPATH=. python section2_pose/evaluation/calibration_synthetic.py
PYTHONPATH=. python section2_pose/evaluation/synthetic_pose_eval.py --n 100 --envelope-trials 100
PYTHONPATH=. python section2_pose/evaluation/uncertainty_mc.py
PYTHONPATH=. python section2_pose/evaluation/plot_results.py
```

For the real Section 1 runtime, pass the extracted G0/G1 coordinates to:

```python
from section2_pose.src.assessment import assess_g0g1

result = assess_g0g1(g0, g1, camera_model, pallet_model)
```

The returned object is deliberately structured for downstream Section 3/4 consumption: a pose is present only when the estimator considers it reliable; otherwise `pose` is null and a machine-readable failure reason is returned.
