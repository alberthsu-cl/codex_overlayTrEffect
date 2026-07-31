# Effect Refinement Prompt Index

This file is the stable entry point for candidate refinement. The iteration
request selects the smaller prompt files needed for the current transition.
Read every prompt file listed by the request, then read the JSON diagnostics
and latest comparison artifacts. Follow the base refinement contract and the
selected transition-family guidance; do not load unrelated family guidance.
