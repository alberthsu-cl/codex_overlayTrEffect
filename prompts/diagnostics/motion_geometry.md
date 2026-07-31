# Motion Geometry Diagnostics

Compare reference and candidate `motion_geometry` before editing. Use
`translation_field` for signed position, `rotation_field` and rotation
direction for signed angle, `radial_scale_field` for scale,
`reflection_or_flip` for handedness, and `spatial_displacement` for residual
region motion. Change the shader code controlling the mismatching transform;
do not use timing, blur, or blend as a substitute.

When `foreground_body_transform` is estimated with adequate confidence, use it
as the preferred body-level measurement for affine candidates. It is based on
feature matches and includes reprojection confidence; ignore it when marked
`low_confidence` or `needs_review`.

For a transform-like transition, use a 2D similarity or affine model when the
evidence supports it. Keep angle, scale, pivot, and translation independently
controlled for outgoing and incoming phases. Preserve zero displacement at
stable endpoints.
