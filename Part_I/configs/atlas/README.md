# N340 atlas routing

`N340_chart_routing.csv` retains the routing metadata from the tracked
production plan while removing checkpoint and output-directory fields that
were tied to a prior machine. Runtime routing is implemented by
`code/src/scripts/gep/selection/solve_blumen_exact_joint_gep_v3.py`, notably
`normalize_plan` and `route_chart`.

The production checkpoint corresponding to a `chart_id` is expected at
`models_saved/production/atlas/N340/<chart_id>/model_state.pt`.
