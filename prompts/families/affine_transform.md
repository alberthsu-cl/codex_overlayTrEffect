# Affine Transform Family

The target is a 2D transform, not a 3D model, unless the analysis explicitly
states otherwise. Implement one explicit transform path:

```text
local = input - pivot - translation
rotated = rotate(local, signed_angle)
scaled = rotated / scale
uv = scaled + pivot
```

Use separate outgoing and incoming phase values. Match the observed signed
rotation, scale envelope, pivot, and translation. A black midpoint is usually
an out-of-bounds or near-zero-scale result, so do not use it as evidence for
additional blur or blend.
