# Frozen solver API inventory

- Solver: `code/scripts/evaluation/mstab17_supersonic_solver_a0ebbf85ec.py`
- SHA-256: `5a5a3169842db15eff63e7ab9d3e9aae6d962f0e2f157e26962ede03425989a0`

## Top-level functions

- Line 492: `def build_parser() -> argparse.ArgumentParser:`
- Line 515: `def main() -> None:`

## Classes and methods

### `Mstab17SupersonicResult` — line 36


### `Mstab17SupersonicSolver` — line 53

- Line 54: `def __init__( self, alpha: float, Mach: float, *, match_y: float = 1.0, ln_p_start_left: float = -5.0, rtol: float = 1e-10, atol: float = 1e-12, max_step: float = float("inf"), min_y_limit: float = 10.0, max_y_limit: float = 80.0, y_limit_factor: float = 4.0, amplitude_match_y: float | None = None, use_mapping: bool = False, mapping_scale: float = 5.0, ln_p_right_min: float = -15.0, ln_p_right_max: float = 5.0, ):`
- Line 97: `def base_velocity(y: np.ndarray | float) -> np.ndarray | float:`
- Line 101: `def base_velocity_derivative(y: np.ndarray | float) -> np.ndarray | float:`
- Line 105: `def phase_speed(self, cr: float, ci: float) -> complex:`
- Line 108: `def asymptotic_gammas(self, cr: float, ci: float) -> tuple[complex, complex]:`
- Line 126: `def estimate_y_limit(self, cr: float, ci: float) -> float:`
- Line 133: `def xi_to_y(self, xi: np.ndarray | float) -> np.ndarray | float:`
- Line 136: `def dy_dxi(self, xi: np.ndarray | float) -> np.ndarray | float:`
- Line 140: `def y_to_xi(self, y: float) -> float:`
- Line 145: `def riccati_system_real_split( self, y: float, state: np.ndarray, cr: float, ci: float, ) -> list[float]:`
- Line 169: `def riccati_system_real_split_xi( self, xi: float, state: np.ndarray, cr: float, ci: float, ) -> list[float]:`
- Line 181: `def get_trajectories( self, cr: float, ci: float, ln_p_start_right: float = -5.0, ) -> tuple[solve_ivp, solve_ivp, solve_ivp, float]:`
- Line 284: `def _interp_component(y_target: float, solution: solve_ivp, component_index: int) -> float:`
- Line 292: `def stage1_mismatch(self, cr: float, ci: float) -> float:`
- Line 305: `def solve_eigenvalue( self, *, cr_min: float = 0.03, cr_max: float = 0.35, ci_min: float = 0.01, ci_max: float = 0.12, max_iter: int = 12, tol: float = 1e-7, grid_size: int = 4, constrain_to_initial_box: bool = False, ) -> tuple[float, float, float]:`
- Line 370: `def stage2_objective(self, ln_p_start_right: float, cr: float, ci: float) -> float:`
- Line 379: `def solve( self, *, cr_min: float = 0.03, cr_max: float = 0.35, ci_min: float = 0.01, ci_max: float = 0.12, max_iter: int = 12, tol: float = 1e-7, grid_size: int = 4, constrain_to_initial_box: bool = False, ) -> Mstab17SupersonicResult:`
- Line 428: `def plot_mode(self, result: Mstab17SupersonicResult, output_path: Path | None = None) -> None:`

## Existing scripts referring to the solver

### `code/scripts/data_preparation/prepare_build_supersonic_fixed_ci_shooting_anchors.py`

- CLI arguments: `--ci-datasets`, `--cr-datasets`, `--cr-max`, `--cr-min`, `--n-grid`, `--output`

### `code/scripts/data_preparation/prepare_build_supersonic_fixed_ci_shooting_extension_final_6f151a02f3.py`

- CLI arguments: `--anchors`, `--eigen-output`, `--fields-output`, `--mapping-scale`, `--match-y`, `--max-y-limit`, `--min-y-limit`, `--y-limit-factor`

### `code/scripts/evaluation/freeze_supersonic_v2_assets_and_code.py`

- CLI arguments detected: none

### `code/scripts/evaluation/package_supersonic_sparse_reference_v2_final_freeze.py`

- CLI arguments detected: none

### `code/scripts/shooting/solve_refine_M18_M19_near_valid_branch.py`

- CLI arguments: `--alpha-step`, `--ci-width`, `--cr-width`, `--final-y-limit`, `--grid-size`, `--mapping-scale`, `--max-iter`, `--output-dir`, `--search-y-limit`, `--seed-csv`

### `code/scripts/audits/audit_scan_supersonic_M18_M19_strict_modal_validation.py`

- CLI arguments: `--alpha-max`, `--alpha-min`, `--alpha-step`, `--grid-size`, `--machs`, `--max-adjacent-jump`, `--max-center-jump`, `--max-edge-frac`, `--max-iter`, `--max-validate-per-point`, `--max-ylimit-rel-l2`, `--output-dir`, `--search-y-limit`, `--stage1-tol`, `--stage2-tol`, `--task-id`

### `code/scripts/audits/audit_M18_M19_max_step_convergence.py`

- CLI arguments: `--candidate-csv`, `--mapping-scale`, `--max-y-limit`, `--n-each`, `--output-dir`, `--representative`

### `code/scripts/audits/audit_M18_M19_refined_convergence.py`

- CLI arguments: `--candidate-csv`, `--ci-width`, `--cr-width`, `--grid-size`, `--max-iter`, `--max-points`, `--output-dir`

### `code/scripts/audits/audit_supersonic_ci_first_shooting_mismatch_map.py`

- CLI arguments: `--alpha`, `--ci-max`, `--ci-min`, `--cr-max`, `--cr-min`, `--mach`, `--n-ci`, `--n-cr`, `--output`

### `code/scripts/audits/audit_supersonic_shooting_visual_validation.py`

- CLI arguments: `--min-half-width`, `--output-stem`, `--summary-csv`, `--threshold-ratio`

