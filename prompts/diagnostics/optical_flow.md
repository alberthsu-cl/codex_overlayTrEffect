# Optical Flow Diagnostics

Optical flow is evidence, not ground truth. It becomes unreliable when the
image is black, low-texture, heavily scaled, or changes source identity.
Ignore tiny fragments and low-confidence estimates. Do not infer a fixed
number of regions or direction buckets from noisy components.

When a visible foreground body or card can be detected, prefer its fitted
translation, rotation, and scale over a global flow average. Otherwise use
the reliable flow vectors and report uncertainty.
