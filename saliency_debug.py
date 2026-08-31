import numpy as np
import os
import sys

PROJECT_ROOT = r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from individual_model_comparison import compute_auditory_saliency

np.random.seed(0)
noise = np.abs(np.random.randn(64, 128)).astype(np.float32)
noise_sal, _ = compute_auditory_saliency(noise, N=3)

event = noise.copy()
event[28:36, 60:68] += 8.0
event_sal, _ = compute_auditory_saliency(event, N=3)

print("noise sum:", noise_sal.sum())
print("event sum:", event_sal.sum())
print("noise max:", noise_sal.max())
print("event max:", event_sal.max())
diff = event_sal - noise_sal
print("diff sum:", diff.sum(), "diff max:", diff.max(), "diff min:", diff.min())

