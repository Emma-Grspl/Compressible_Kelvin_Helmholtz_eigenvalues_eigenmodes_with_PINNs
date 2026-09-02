# Modal reconstruction path

## Selected solver definitions

### `base_velocity` — line 97

```python
    def base_velocity(y: np.ndarray | float) -> np.ndarray | float:
        return np.tanh(y)
```

### `base_velocity_derivative` — line 101

```python
    def base_velocity_derivative(y: np.ndarray | float) -> np.ndarray | float:
        exp_term = np.exp(-2.0 * np.abs(y))
        return 4.0 * exp_term / (1.0 + exp_term) ** 2
```

### `phase_speed` — line 105

```python
    def phase_speed(self, cr: float, ci: float) -> complex:
        return max(float(cr), 0.0) + 1j * max(float(ci), self.ci_floor)
```

### `asymptotic_gammas` — line 108

```python
    def asymptotic_gammas(self, cr: float, ci: float) -> tuple[complex, complex]:
        """
        Branches decroissantes dans les far fields U -> -1 et U -> +1.
        """
        c = self.phase_speed(cr, ci)
        r_inf_left = 1.0 - self.Mach**2 * ((-1.0 - c) ** 2)
        r_inf_right = 1.0 - self.Mach**2 * ((1.0 - c) ** 2)

        gamma_left = self.alpha * _principal_sqrt(r_inf_left)
        if gamma_left.real < 0:
            gamma_left = -gamma_left

        gamma_right = -self.alpha * _principal_sqrt(r_inf_right)
        if gamma_right.real > 0:
            gamma_right = -gamma_right

        return gamma_left, gamma_right
```

### `get_trajectories` — line 181

```python
    def get_trajectories(
        self,
        cr: float,
        ci: float,
        ln_p_start_right: float = -5.0,
    ) -> tuple[solve_ivp, solve_ivp, solve_ivp, float]:
        """
        Retourne :
        - branche gauche jusqu'a match_y
        - branche droite evaluee a match_y
        - branche droite complete jusqu'a 0
        - taille de domaine utilisee
        """
        gamma_left_inf, gamma_right_inf = self.asymptotic_gammas(cr, ci)
        y_limit = self.estimate_y_limit(cr, ci)

        init_left = [gamma_left_inf.real, gamma_left_inf.imag, self.ln_p_start_left, 0.0]
        init_right = [gamma_right_inf.real, gamma_right_inf.imag, ln_p_start_right, 0.0]

        if self.use_mapping:
            xi_left = self.y_to_xi(-y_limit)
            xi_right = self.y_to_xi(y_limit)
            xi_match = self.y_to_xi(self.match_y)
            xi_center = self.y_to_xi(0.0)
            xi_eval_left = np.linspace(xi_left, xi_match, 2500)
            xi_eval_right = np.linspace(xi_right, xi_center, 2500)

            sol_left = solve_ivp(
                self.riccati_system_real_split_xi,
                (xi_left, xi_match),
                init_left,
                t_eval=xi_eval_left,
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            sol_right_match = solve_ivp(
                self.riccati_system_real_split_xi,
                (xi_right, xi_match),
                init_right,
                t_eval=np.array([xi_match]),
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            sol_right_full = solve_ivp(
                self.riccati_system_real_split_xi,
                (xi_right, xi_center),
                init_right,
                t_eval=xi_eval_right,
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            sol_left.t = self.xi_to_y(sol_left.t)
            sol_right_match.t = self.xi_to_y(sol_right_match.t)
            sol_right_full.t = self.xi_to_y(sol_right_full.t)
        else:
            y_eval_left = np.linspace(-y_limit, self.match_y, 2500)
            y_eval_right = np.linspace(y_limit, 0.0, 2500)

            sol_left = solve_ivp(
                self.riccati_system_real_split,
                (-y_limit, self.match_y),
                init_left,
                t_eval=y_eval_left,
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            sol_right_match = solve_ivp(
                self.riccati_system_real_split,
                (y_limit, self.match_y),
                init_right,
                t_eval=np.array([self.match_y]),
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            sol_right_full = solve_ivp(
                self.riccati_system_real_split,
                (y_limit, 0.0),
                init_right,
                t_eval=y_eval_right,
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
        return sol_left, sol_right_match, sol_right_full, y_limit
```

### `_interp_component` — line 284

```python
    def _interp_component(y_target: float, solution: solve_ivp, component_index: int) -> float:
        t = np.asarray(solution.t)
        values = np.asarray(solution.y[component_index])
        if t[0] > t[-1]:
            t = t[::-1]
            values = values[::-1]
        return float(np.interp(y_target, t, values))
```

### `solve` — line 379

```python
    def solve(
        self,
        *,
        cr_min: float = 0.03,
        cr_max: float = 0.35,
        ci_min: float = 0.01,
        ci_max: float = 0.12,
        max_iter: int = 12,
        tol: float = 1e-7,
        grid_size: int = 4,
        constrain_to_initial_box: bool = False,
    ) -> Mstab17SupersonicResult:
        cr_star, ci_star, stage1_err = self.solve_eigenvalue(
            cr_min=cr_min,
            cr_max=cr_max,
            ci_min=ci_min,
            ci_max=ci_max,
            max_iter=max_iter,
            tol=tol,
            grid_size=grid_size,
            constrain_to_initial_box=constrain_to_initial_box,
        )
        amp_opt = minimize_scalar(
            lambda ln_p_right: self.stage2_objective(ln_p_right, cr_star, ci_star),
            bounds=(self.ln_p_right_min, self.ln_p_right_max),
            method="bounded",
        )
        stage2_err = float(amp_opt.fun)
        _, _, _, y_limit = self.get_trajectories(cr_star, ci_star, ln_p_start_right=float(amp_opt.x))
        spectral_success = bool(stage1_err < 5e-2)
        mode_success = bool(stage2_err < 1e-2)
        success = bool(spectral_success and mode_success)
        return Mstab17SupersonicResult(
            alpha=self.alpha,
            Mach=self.Mach,
            cr=cr_star,
            ci=ci_star,
            omega_i=self.alpha * ci_star,
            stage1_mismatch=stage1_err,
            stage2_mismatch=stage2_err,
            y_limit=y_limit,
            ln_p_start_right=float(amp_opt.x),
            spectral_success=spectral_success,
            mode_success=mode_success,
            success=success,
            use_mapping=self.use_mapping,
            mapping_scale=self.mapping_scale,
        )
```

### `plot_mode` — line 428

```python
    def plot_mode(self, result: Mstab17SupersonicResult, output_path: Path | None = None) -> None:
        sol_left, _, sol_right_full, _ = self.get_trajectories(
            result.cr,
            result.ci,
            ln_p_start_right=result.ln_p_start_right,
        )

        y_left = np.asarray(sol_left.t)
        y_right = np.asarray(sol_right_full.t)
        k_left, q_left, ln_p_left, phi_left = sol_left.y
        k_right, q_right, ln_p_right, phi_right = sol_right_full.y

        abs_p_left = np.exp(ln_p_left)
        abs_p_right = np.exp(ln_p_right)

        phi_left_0 = self._interp_component(0.0, sol_left, 3)
        phi_right_0 = self._interp_component(0.0, sol_right_full, 3)
        phase_shift = phi_left_0 - phi_right_0

        mode_left = abs_p_left * np.cos(phi_left)
        mode_right = abs_p_right * np.cos(phi_right + phase_shift)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].plot(y_left, abs_p_left, label="Left")
        axes[0, 0].plot(y_right, abs_p_right, "--", label="Right")
        axes[0, 0].set_title(r"Amplitude $|p|$")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(y_left, k_left, label=r"$\kappa$ left")
        axes[0, 1].plot(y_left, q_left, ":", label=r"$q$ left")
        axes[0, 1].plot(y_right, k_right, "--", label=r"$\kappa$ right")
        axes[0, 1].plot(y_right, q_right, "-.", label=r"$q$ right")
        axes[0, 1].set_title("Variables de Riccati")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[1, 0].plot(y_left, mode_left, label="Left")
        axes[1, 0].plot(y_right, mode_right, "--", label="Right")
        axes[1, 0].set_title(r"Mode physique $\Re(p)$")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        yy = np.linspace(-result.y_limit, result.y_limit, 400)
        axes[1, 1].plot(yy, np.tanh(yy), label=r"$U(y)=\tanh(y)$")
        axes[1, 1].axhline(result.cr, color="orange", linestyle="--", label=r"$c_r$")
        axes[1, 1].set_title("Profil moyen")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        fig.suptitle(
            f"mstab17 supersonic | alpha={self.alpha:.3f}, M={self.Mach:.3f}, "
            f"cr={result.cr:.5f}, ci={result.ci:.5f}, omega_i={result.omega_i:.5f}, "
            f"mapping={'on' if result.use_mapping else 'off'}"
        )
        fig.tight_layout()
        if output_path is None:
            plt.show()
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
```

## Existing field-export code

### `code/scripts/data_preparation/prepare_build_final_sparse_reference_v2_with_smallM_campaign.py`

Lines 14–29

```python
0014: BASE = Path("assets/classic_supersonic/final_sparse_PINN_reference")
0015: CAMPAIGN = Path("assets/classic_supersonic/campaign_smallM_low_high_alpha_scan")
0016: OUT = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_smallM_M18M19")
0017: OUT.mkdir(parents=True, exist_ok=True)
0018: 
0019: BASE_SPEC = BASE / "supersonic_sparse_PINN_reference_spectral.csv"
0020: BASE_FIELDS = BASE / "supersonic_sparse_PINN_reference_modal_fields.csv"
0021: 
0022: CAMPAIGN_SPEC = CAMPAIGN / "campaign_near_valid_best_per_target.csv"
0023: CAMPAIGN_FIELDS = CAMPAIGN / "campaign_smallM_modal_fields_reconstructed.csv"
0024: 
0025: BLUMEN_CI = Path("assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv")
0026: 
0027: 
0028: def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
0029:     if "Mach" not in df.columns and "M" in df.columns:
```

Lines 401–413

```python
0401: fig.tight_layout()
0402: 
0403: # -------------------------------------------------------------------
0404: # Write outputs.
0405: # -------------------------------------------------------------------
0406: spec_out = OUT / "supersonic_sparse_PINN_reference_v2_spectral.csv"
0407: fields_out = OUT / "supersonic_sparse_PINN_reference_v2_modal_fields.csv"
0408: coverage_out = OUT / "coverage_by_Mach_v2.csv"
0409: suggested_out = OUT / "suggested_remaining_targets_v2.csv"
0410: overlay_png = OUT / "blumen_ci_overlay_sparse_PINN_reference_v2.png"
0411: overlay_pdf = OUT / "blumen_ci_overlay_sparse_PINN_reference_v2.pdf"
0412: 
0413: spec_final.to_csv(spec_out, index=False)
```

Lines 426–438

```python
0426:     "n_total_modal_rows": int(len(fields_final)),
0427:     "point_counts_by_Mach": spec_final.groupby("Mach").size().to_dict(),
0428:     "validation_status_counts": spec_final["validation_status"].value_counts(dropna=False).to_dict(),
0429:     "boundary_flag_points": int(new_campaign["boundary_flag"].sum()),
0430:     "outputs": {
0431:         "spectral": str(spec_out),
0432:         "modal_fields": str(fields_out),
0433:         "coverage": str(coverage_out),
0434:         "suggested_remaining_targets": str(suggested_out),
0435:         "overlay_png": str(overlay_png),
0436:         "overlay_pdf": str(overlay_pdf),
0437:     },
0438:     "important_note": (
```

### `code/scripts/data_preparation/prepare_build_final_sparse_supersonic_reference_with_M18_M19.py`

Lines 19–32

```python
0019: BASE_SPECTRAL_CANDIDATES = [
0020:     Path("assets/classic_supersonic/supersonic_modal_spectral_validated.csv"),
0021:     Path("assets/classic_supersonic/final_44pts_validated_only/supersonic_modal_spectral_validated_44pts.csv"),
0022: ]
0023: 
0024: BASE_FIELDS_CANDIDATES = [
0025:     Path("assets/classic_supersonic/final_44pts_validated_only/supersonic_modal_fields_p_rho_u_v_44pts_validated_only.csv"),
0026:     Path("assets/classic_supersonic/validated_modal_points/rebuilt_aggregates_latest/supersonic_reference_core_local_modal_fields_REBUILT.csv"),
0027: ]
0028: 
0029: CONV = Path("assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/refined_near_valid_branch/convergence_audit_full")
0030: CONV_AUDIT = CONV / "convergence_audit_by_point.csv"
0031: CORE_TAIL = CONV / "core_vs_tail_phase_convergence.csv"
0032: FIELDS_BY_SETTING = CONV / "fields_by_setting"
```

Lines 324–336

```python
0324:     .drop(columns=["_Mach_key", "_alpha_key", "_y_key"])
0325:     .sort_values(["Mach", "alpha", "y"])
0326:     .reset_index(drop=True)
0327: )
0328: 
0329: spectral_out = OUT / "supersonic_sparse_PINN_reference_spectral.csv"
0330: fields_out = OUT / "supersonic_sparse_PINN_reference_modal_fields.csv"
0331: 
0332: spectral_final.to_csv(spectral_out, index=False)
0333: fields_final.to_csv(fields_out, index=False)
0334: 
0335: 
0336: # ------------------------------------------------------------------
```

Lines 485–497

```python
0485: # ------------------------------------------------------------------
0486: # 6. Summary.
0487: # ------------------------------------------------------------------
0488: summary = {
0489:     "status": "sparse_PINN_reference_built",
0490:     "spectral_file": str(spectral_out),
0491:     "modal_fields_file": str(fields_out),
0492:     "overlay_png": str(overlay_png),
0493:     "overlay_pdf": str(overlay_pdf),
0494:     "coverage_file": str(coverage_out),
0495:     "suggested_next_targets_file": str(suggested_out),
0496:     "base_spectral_source": str(base_spectral_path),
0497:     "base_fields_source": str(base_fields_path),
```

### `code/scripts/data_preparation/prepare_build_supersonic_fixed_ci_shooting_extension_final_6f151a02f3.py`

Lines 49–69

```python
0049:             mapping_scale=mapping_scale,
0050:             min_y_limit=min_y_limit,
0051:             max_y_limit=max_y_limit,
0052:             y_limit_factor=y_limit_factor,
0053:         )
0054: 
0055:         sol_left, _, sol_right_full, y_limit = solver.get_trajectories(
0056:             cr, ci, ln_p_start_right=0.0
0057:         )
0058: 
0059:         out = row.to_dict()
0060: 
0061:         if not (sol_left.success and sol_right_full.success):
0062:             out.update({
0063:                 "ln_p_start_right_exact": np.nan,
0064:                 "ln_p_left_target": np.nan,
0065:                 "ln_p_right_target_if_start_0": np.nan,
0066:                 "stage2_mismatch_exact": np.nan,
0067:                 "mode_success_exact": False,
0068:                 "full_success_exact": False,
0069:                 "y_limit_exact": float(y_limit),
```

Lines 82–94

```python
0082: 
0083:         spectral_success = parse_bool(row.get("spectral_success", False))
0084:         mode_success_exact = bool(stage2_exact < 1e-2)
0085:         full_success_exact = bool(spectral_success and mode_success_exact)
0086: 
0087:         out.update({
0088:             "ln_p_start_right_exact": ln_required,
0089:             "ln_p_left_target": float(ln_left),
0090:             "ln_p_right_target_if_start_0": float(ln_right_zero),
0091:             "stage2_mismatch_exact": stage2_exact,
0092:             "mode_success_exact": mode_success_exact,
0093:             "full_success_exact": full_success_exact,
0094:             "y_limit_exact": float(y_limit),
```

Lines 107–119

```python
0107:             f"stage2={stage2_exact} full_success={full_success_exact}"
0108:         )
0109: 
0110:     return pd.DataFrame(rows)
0111: 
0112: 
0113: def build_modal_fields(
0114:     enriched: pd.DataFrame,
0115:     *,
0116:     match_y: float,
0117:     mapping_scale: float,
0118:     min_y_limit: float,
0119:     max_y_limit: float,
```

Lines 123–144

```python
0123: 
0124:     for _, row in enriched.iterrows():
0125:         mach = float(row["Mach"])
0126:         alpha = float(row["alpha"])
0127:         cr = float(row["shooting_cr"])
0128:         ci = float(row["shooting_ci"])
0129:         ln_p = float(row["ln_p_start_right_exact"])
0130: 
0131:         print(f"[modal-fields] M={mach} alpha={alpha} c=({cr}, {ci}) ln_p={ln_p}")
0132: 
0133:         fields = reconstruct_shooting_fields(
0134:             alpha=alpha,
0135:             mach=mach,
0136:             cr=cr,
0137:             ci=ci,
0138:             ln_p_start_right=ln_p,
0139:             match_y=match_y,
0140:             use_mapping=True,
0141:             mapping_scale=mapping_scale,
0142:             min_y_limit=min_y_limit,
0143:             max_y_limit=max_y_limit,
0144:             y_limit_factor=y_limit_factor,
```

Lines 149–170

```python
0149:             rows.append({
0150:                 "Mach": mach,
0151:                 "alpha": alpha,
0152:                 "cr": cr,
0153:                 "ci": ci,
0154:                 "omega_i": alpha * ci,
0155:                 "ln_p_start_right_exact": ln_p,
0156:                 "y": float(y[i]),
0157:                 "rho_real": float(np.real(fields["rho"][i])),
0158:                 "rho_imag": float(np.imag(fields["rho"][i])),
0159:                 "u_real": float(np.real(fields["u"][i])),
0160:                 "u_imag": float(np.imag(fields["u"][i])),
0161:                 "v_real": float(np.real(fields["v"][i])),
0162:                 "v_imag": float(np.imag(fields["v"][i])),
0163:                 "p_real": float(np.real(fields["p"][i])),
0164:                 "p_imag": float(np.imag(fields["p"][i])),
0165:                 "source": "fixed_ci_shooting_exact_amplitude",
0166:                 "validation_status": str(row["final_status"]),
0167:             })
0168: 
0169:     return pd.DataFrame(rows)
0170: 
```

Lines 181–193

```python
0181:         type=Path,
0182:         default=OUTDIR / "supersonic_neutral_M180_M190_fixed_ci_shooting_anchors_exact_amplitude.csv",
0183:     )
0184:     parser.add_argument(
0185:         "--fields-output",
0186:         type=Path,
0187:         default=OUTDIR / "supersonic_neutral_M180_M190_fixed_ci_shooting_modal_fields.csv",
0188:     )
0189:     parser.add_argument("--match-y", type=float, default=1.0)
0190:     parser.add_argument("--mapping-scale", type=float, default=5.0)
0191:     parser.add_argument("--min-y-limit", type=float, default=10.0)
0192:     parser.add_argument("--max-y-limit", type=float, default=1200.0)
0193:     parser.add_argument("--y-limit-factor", type=float, default=10.0)
```

Lines 204–226

```python
0204:         y_limit_factor=args.y_limit_factor,
0205:     )
0206:     args.eigen_output.parent.mkdir(parents=True, exist_ok=True)
0207:     enriched.to_csv(args.eigen_output, index=False)
0208:     print(f"[final] wrote {args.eigen_output}")
0209: 
0210:     fields = build_modal_fields(
0211:         enriched,
0212:         match_y=args.match_y,
0213:         mapping_scale=args.mapping_scale,
0214:         min_y_limit=args.min_y_limit,
0215:         max_y_limit=args.max_y_limit,
0216:         y_limit_factor=args.y_limit_factor,
0217:     )
0218:     args.fields_output.parent.mkdir(parents=True, exist_ok=True)
0219:     fields.to_csv(args.fields_output, index=False)
0220:     print(f"[final] wrote {args.fields_output}")
0221:     print("[final] rows =", len(fields))
0222:     print(fields.groupby(["Mach", "alpha"]).size())
0223: 
0224: 
0225: if __name__ == "__main__":
0226:     main()
```

### `code/scripts/evaluation/freeze_supersonic_sparse_PINN_reference_v2.py`

Lines 15–36

```python
0015: 
0016: 
0017: SRC = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_smallM_M18M19")
0018: OUT = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_CONFIRMED")
0019: 
0020: SPEC_IN = SRC / "supersonic_sparse_PINN_reference_v2_spectral.csv"
0021: FIELDS_IN = SRC / "supersonic_sparse_PINN_reference_v2_modal_fields.csv"
0022: OVERLAY_IN = SRC / "blumen_ci_overlay_sparse_PINN_reference_v2.png"
0023: COVERAGE_IN = SRC / "coverage_by_Mach_v2.csv"
0024: SUGGESTED_IN = SRC / "suggested_remaining_targets_v2.csv"
0025: SUMMARY_IN = SRC / "summary.json"
0026: 
0027: OUT.mkdir(parents=True, exist_ok=True)
0028: 
0029: SPEC_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_spectral.csv"
0030: FIELDS_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv"
0031: POINT_AUDIT_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_point_audit.csv"
0032: 
0033: PDF_OVERVIEW = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modes_overview.pdf"
0034: PDF_CORE_TAIL = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modes_core_tail.pdf"
0035: 
0036: README_OUT = OUT / "README_FREEZE.md"
```

Lines 51–66

```python
0051:         if c in df.columns:
0052:             return c
0053:     raise KeyError(f"Missing any of columns: {candidates}")
0054: 
0055: 
0056: FIELD_COLS = {
0057:     "p": (["p_real", "p_re", "Re_p"], ["p_imag", "p_im", "Im_p"]),
0058:     "rho": (["rho_real", "rho_re", "Re_rho"], ["rho_imag", "rho_im", "Im_rho"]),
0059:     "u": (["u_real", "u_re", "Re_u"], ["u_imag", "u_im", "Im_u"]),
0060:     "v": (["v_real", "v_re", "Re_v"], ["v_imag", "v_im", "Im_v"]),
0061: }
0062: 
0063: 
0064: def complex_field(df: pd.DataFrame, name: str) -> np.ndarray:
0065:     re_candidates, im_candidates = FIELD_COLS[name]
0066:     re_col = find_col(df, re_candidates)
```

Lines 373–385

```python
0373:     "n_points_with_fields": int(point_audit["has_fields"].sum()),
0374:     "n_points_missing_fields": int((~point_audit["has_fields"]).sum()),
0375:     "point_counts_by_Mach": spec.groupby("Mach").size().to_dict(),
0376:     "validation_status_counts": spec["validation_status"].astype(str).value_counts(dropna=False).to_dict(),
0377:     "outputs": {
0378:         "spectral": str(SPEC_OUT),
0379:         "modal_fields": str(FIELDS_OUT),
0380:         "point_audit": str(POINT_AUDIT_OUT),
0381:         "modes_overview_pdf": str(PDF_OVERVIEW),
0382:         "modes_core_tail_pdf": str(PDF_CORE_TAIL),
0383:         "readme": str(README_OUT),
0384:         "sha256": str(SHA_OUT),
0385:     },
```

### `code/scripts/evaluation/freeze_supersonic_v2_assets_and_code.py`

Lines 25–49

```python
0025: DATA = OUT / "data"
0026: 
0027: for d in [OUT, ASSETS, CODE, REPORTS, DATA]:
0028:     d.mkdir(parents=True, exist_ok=True)
0029: 
0030: RAW_SPEC = RAW / "supersonic_sparse_PINN_reference_v2_CONFIRMED_spectral.csv"
0031: RAW_FIELDS = RAW / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv"
0032: 
0033: POL_SPEC = POL / "supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_spectral.csv"
0034: POL_FIELDS = POL / "supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_modal_fields.csv"
0035: 
0036: OVERLAY = RAW / "blumen_ci_overlay_sparse_PINN_reference_v2.png"
0037: 
0038: SQUARE_PDF = REPORTS / "supersonic_sparse_PINN_reference_v2_FROZEN_modes_square.pdf"
0039: TAIL_SQUARE_PDF = REPORTS / "supersonic_sparse_PINN_reference_v2_FROZEN_tail_polished_square_review.pdf"
0040: 
0041: SPEC_OUT = DATA / "supersonic_sparse_PINN_reference_v2_FROZEN_spectral.csv"
0042: FIELDS_RAW_OUT = DATA / "supersonic_sparse_PINN_reference_v2_FROZEN_modal_fields_raw_confirmed.csv"
0043: FIELDS_POL_OUT = DATA / "supersonic_sparse_PINN_reference_v2_FROZEN_modal_fields_tail_polished_v1.csv"
0044: 
0045: MANIFEST = OUT / "manifest.json"
0046: SHA = OUT / "SHA256SUMS.txt"
0047: README = OUT / "README.md"
0048: 
0049: 
```

### `code/scripts/evaluation/package_supersonic_sparse_reference_v2_final_freeze.py`

Lines 31–49

```python
0031: SPEC_CANDIDATES = [
0032:     FROZEN / "data/supersonic_sparse_PINN_reference_v2_FROZEN_spectral.csv",
0033:     ROOT / "final_sparse_PINN_reference_v2_CONFIRMED/supersonic_sparse_PINN_reference_v2_CONFIRMED_spectral.csv",
0034: ]
0035: 
0036: RAW_FIELDS_CANDIDATES = [
0037:     FROZEN / "data/supersonic_sparse_PINN_reference_v2_FROZEN_modal_fields_raw_confirmed.csv",
0038:     ROOT / "final_sparse_PINN_reference_v2_CONFIRMED/supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv",
0039: ]
0040: 
0041: POL_FIELDS_CANDIDATES = [
0042:     FROZEN / "data/supersonic_sparse_PINN_reference_v2_FROZEN_modal_fields_tail_polished_v1.csv",
0043:     ROOT / "final_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1/supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_modal_fields.csv",
0044: ]
0045: 
0046: MODES_PDF_CANDIDATES = [
0047:     FROZEN / "reports/supersonic_sparse_PINN_reference_v2_FROZEN_modes_square_full_y.pdf",
0048:     FROZEN / "reports/supersonic_sparse_PINN_reference_v2_FROZEN_modes_square.pdf",
0049:     ROOT / "final_sparse_PINN_reference_v2_CONFIRMED/supersonic_sparse_PINN_reference_v2_CONFIRMED_modes_overview.pdf",
```

Lines 400–413

```python
0400: 
0401:     spec = norm_cols(pd.read_csv(spec_src))
0402:     spec = spec.sort_values(["Mach", "alpha"]).drop_duplicates(["Mach", "alpha"], keep="first").reset_index(drop=True)
0403: 
0404:     # Data canonical copies.
0405:     spectral_out = DATA_DIR / "supersonic_sparse_PINN_reference_v2_FINAL_spectral.csv"
0406:     raw_out = DATA_DIR / "supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_raw_confirmed.csv"
0407:     pol_out = DATA_DIR / "supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_tail_polished_v1.csv"
0408: 
0409:     spec.to_csv(spectral_out, index=False)
0410:     shutil.copy2(raw_src, raw_out)
0411:     shutil.copy2(pol_src, pol_out)
0412: 
0413:     # Modes PDFs.
```

### `code/scripts/evaluation/polish_supersonic_v2_left_tails.py`

Lines 15–30

```python
0015: 
0016: SRC = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_CONFIRMED")
0017: OUT = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1")
0018: OUT.mkdir(parents=True, exist_ok=True)
0019: 
0020: SPEC_IN = SRC / "supersonic_sparse_PINN_reference_v2_CONFIRMED_spectral.csv"
0021: FIELDS_IN = SRC / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv"
0022: 
0023: SPEC_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_spectral.csv"
0024: FIELDS_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_modal_fields.csv"
0025: AUDIT_OUT = OUT / "tail_polish_audit.csv"
0026: PDF_OUT = OUT / "tail_polish_raw_vs_polished_left_tail_review.pdf"
0027: SUMMARY_OUT = OUT / "summary_tail_polished_v1.json"
0028: SHA_OUT = OUT / "SHA256SUMS.txt"
0029: 
0030: FIELDS = ["p", "rho", "u", "v"]
```

Lines 369–381

```python
0369:             "Fit complex exponential in weak left-tail overlap using p, then apply same lambda "
0370:             "to p/rho/u/v for y < y_join. Core is unchanged."
0371:         ),
0372:     },
0373:     "outputs": {
0374:         "spectral": str(SPEC_OUT),
0375:         "modal_fields": str(FIELDS_OUT),
0376:         "audit": str(AUDIT_OUT),
0377:         "review_pdf": str(PDF_OUT),
0378:     },
0379:     "important_note": (
0380:         "This is a tail-regularized export for PINN/visual use. "
0381:         "The raw confirmed dataset remains the primary frozen reference."
```

### `code/scripts/shooting/solve_refine_M18_M19_near_valid_branch.py`

Lines 138–173

```python
0138: 
0139:     df1200 = reconstruct_dataframe(
0140:         alpha=alpha,
0141:         mach=mach,
0142:         cr=cr,
0143:         ci=ci,
0144:         ln_p_start_right=ln1200,
0145:         max_y_limit=1200.0,
0146:         mapping_scale=mapping_scale,
0147:     )
0148: 
0149:     df_final = reconstruct_dataframe(
0150:         alpha=alpha,
0151:         mach=mach,
0152:         cr=cr,
0153:         ci=ci,
0154:         ln_p_start_right=ln_final,
0155:         max_y_limit=final_y_limit,
0156:         mapping_scale=mapping_scale,
0157:     )
0158: 
0159:     shape = field_shape_metrics(df_final)
0160:     stab = ylimit_stability(df1200, df_final)
0161: 
0162:     metrics = {
0163:         "stage1_mismatch": float(stage1),
0164:         "stage2_mismatch_exact_1200": float(stage2_1200),
0165:         "stage2_mismatch_exact_final": float(stage2_final),
0166:         "ln_p_start_right_exact_1200": float(ln1200),
0167:         "ln_p_start_right_exact_final": float(ln_final),
0168:         "y_limit_1200": float(ylimit1200),
0169:         "y_limit_final": float(ylimit_final),
0170:         "amplitude_status_1200": amp_status1200,
0171:         "amplitude_status_final": amp_status_final,
0172:     }
0173:     metrics.update(shape)
```

Lines 324–336

```python
0324:     accepted = out[out["validation_status"].eq("refined_near_valid_requires_visual_confirmation")].copy()
0325:     accepted.to_csv(args.output_dir / "refined_near_valid_candidates.csv", index=False)
0326: 
0327:     if accepted_fields:
0328:         fields_all = pd.concat(accepted_fields, ignore_index=True)
0329:         fields_all = fields_all.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)
0330:         fields_all.to_csv(args.output_dir / "refined_near_valid_modal_fields.csv", index=False)
0331: 
0332:     summary = {
0333:         "status": "refined_near_valid_branch_requires_visual_confirmation",
0334:         "alpha_step": args.alpha_step,
0335:         "mapping_scale": args.mapping_scale,
0336:         "final_y_limit": args.final_y_limit,
```

### `code/scripts/audits/audit_scan_supersonic_M18_M19_strict_modal_validation.py`

Lines 11–26

```python
0011: 
0012: from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
0013: from scripts.audits.audit_supersonic_shooting_visual_validation_6969b4f1bf import reconstruct_shooting_fields
0014: 
0015: 
0016: FIELDS = {
0017:     "p": ("p_real", "p_imag"),
0018:     "rho": ("rho_real", "rho_imag"),
0019:     "u": ("u_real", "u_imag"),
0020:     "v": ("v_real", "v_imag"),
0021: }
0022: 
0023: 
0024: def exact_log_amplitude(
0025:     *,
0026:     alpha: float,
```

Lines 39–52

```python
0039:         mapping_scale=mapping_scale,
0040:         min_y_limit=10.0,
0041:         max_y_limit=max_y_limit,
0042:         y_limit_factor=10.0,
0043:     )
0044: 
0045:     sol_left, _, sol_right_full, y_limit = solver.get_trajectories(
0046:         cr, ci, ln_p_start_right=0.0
0047:     )
0048: 
0049:     if not (sol_left.success and sol_right_full.success):
0050:         return np.nan, np.nan, float(y_limit), "trajectory_failure"
0051: 
0052:     target_y = solver.amplitude_match_y
```

Lines 70–92

```python
0070: def reconstruct_dataframe(
0071:     *,
0072:     alpha: float,
0073:     mach: float,
0074:     cr: float,
0075:     ci: float,
0076:     ln_p_start_right: float,
0077:     max_y_limit: float,
0078:     match_y: float = 1.0,
0079:     mapping_scale: float = 5.0,
0080: ) -> pd.DataFrame:
0081:     fields = reconstruct_shooting_fields(
0082:         alpha=alpha,
0083:         mach=mach,
0084:         cr=cr,
0085:         ci=ci,
0086:         ln_p_start_right=ln_p_start_right,
0087:         match_y=match_y,
0088:         use_mapping=True,
0089:         mapping_scale=mapping_scale,
0090:         min_y_limit=10.0,
0091:         max_y_limit=max_y_limit,
0092:         y_limit_factor=10.0,
```

Lines 97–109

```python
0097:     out = pd.DataFrame({
0098:         "Mach": mach,
0099:         "alpha": alpha,
0100:         "cr": cr,
0101:         "ci": ci,
0102:         "omega_i": alpha * ci,
0103:         "ln_p_start_right_exact": ln_p_start_right,
0104:         "max_y_limit_used": max_y_limit,
0105:         "y": y,
0106:     })
0107: 
0108:     for name in ["p", "rho", "u", "v"]:
0109:         z = np.asarray(fields[name])
```

Lines 373–389

```python
0373:             cr=cr,
0374:             ci=ci,
0375:             max_y_limit=1200.0,
0376:         )
0377: 
0378:         out.update({
0379:             "ln_p_start_right_exact_600": ln600,
0380:             "stage2_mismatch_exact_600": stage2_600,
0381:             "y_limit_600": y600,
0382:             "amplitude_status_600": amp_status_600,
0383:             "ln_p_start_right_exact_1200": ln1200,
0384:             "stage2_mismatch_exact_1200": stage2_1200,
0385:             "y_limit_1200": y1200,
0386:             "amplitude_status_1200": amp_status_1200,
0387:         })
0388: 
0389:         if not np.isfinite(ln600) or not np.isfinite(ln1200):
```

Lines 393–414

```python
0393: 
0394:         df600 = reconstruct_dataframe(
0395:             alpha=alpha,
0396:             mach=mach,
0397:             cr=cr,
0398:             ci=ci,
0399:             ln_p_start_right=ln600,
0400:             max_y_limit=600.0,
0401:         )
0402: 
0403:         df1200 = reconstruct_dataframe(
0404:             alpha=alpha,
0405:             mach=mach,
0406:             cr=cr,
0407:             ci=ci,
0408:             ln_p_start_right=ln1200,
0409:             max_y_limit=1200.0,
0410:         )
0411: 
0412:         shape = field_shape_metrics(df1200)
0413:         stab = ylimit_stability(df600, df1200)
0414: 
```

### `code/scripts/audits/audit_M18_M19_mapping_vs_box_effect.py`

Lines 20–35

```python
0020:     interp_complex,
0021:     aligned_rel_l2,
0022:     mode_xlim,
0023: )
0024: 
0025: FIELD_SPECS = [
0026:     ("p", "p_real", "p_imag"),
0027:     ("rho", "rho_real", "rho_imag"),
0028:     ("v", "v_real", "v_imag"),
0029:     ("u", "u_real", "u_imag"),
0030: ]
0031: 
0032: ROOT = Path("assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/refined_near_valid_branch")
0033: CAND = ROOT / "refined_near_valid_candidates.csv"
0034: OUT = ROOT / "mapping_vs_box_audit"
0035: FIELDS_DIR = OUT / "fields"
```

Lines 85–97

```python
0085: 
0086:         df = reconstruct_df(
0087:             alpha=a,
0088:             mach=M,
0089:             cr=cr,
0090:             ci=ci,
0091:             ln_p_start_right=ln_amp,
0092:             max_y_limit=y_limit,
0093:             mapping_scale=mapping_scale,
0094:             max_step=float("inf"),
0095:         )
0096: 
0097:         df["audit_setting"] = label
```

### `code/scripts/audits/audit_M18_M19_max_step_convergence.py`

Lines 14–29

```python
0014: 
0015: from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
0016: from scripts.audits.audit_supersonic_shooting_visual_validation_6969b4f1bf import reconstruct_shooting_fields
0017: 
0018: 
0019: FIELD_SPECS = [
0020:     ("p", "p_real", "p_imag"),
0021:     ("rho", "rho_real", "rho_imag"),
0022:     ("v", "v_real", "v_imag"),
0023:     ("u", "u_real", "u_imag"),
0024: ]
0025: 
0026: 
0027: MAX_STEPS = [
0028:     ("inf", float("inf")),
0029:     ("2", 2.0),
```

Lines 64–77

```python
0064:         min_y_limit=10.0,
0065:         max_y_limit=max_y_limit,
0066:         y_limit_factor=10.0,
0067:         max_step=max_step,
0068:     )
0069: 
0070:     sol_left, _, sol_right_full, y_limit = solver.get_trajectories(
0071:         cr, ci, ln_p_start_right=0.0
0072:     )
0073: 
0074:     if not (sol_left.success and sol_right_full.success):
0075:         return np.nan, np.nan, float(y_limit)
0076: 
0077:     target_y = solver.amplitude_match_y
```

Lines 81–99

```python
0081:     ln_required = float(ln_left - ln_right_zero)
0082:     stage2 = float(solver.stage2_objective(ln_required, cr, ci))
0083: 
0084:     return ln_required, stage2, float(y_limit)
0085: 
0086: 
0087: def reconstruct_df(alpha, mach, cr, ci, ln_p_start_right, max_y_limit, mapping_scale, max_step):
0088:     fields = reconstruct_shooting_fields(
0089:         alpha=alpha,
0090:         mach=mach,
0091:         cr=cr,
0092:         ci=ci,
0093:         ln_p_start_right=ln_p_start_right,
0094:         match_y=1.0,
0095:         use_mapping=True,
0096:         mapping_scale=mapping_scale,
0097:         min_y_limit=10.0,
0098:         max_y_limit=max_y_limit,
0099:         y_limit_factor=10.0,
```

Lines 246–258

```python
0246: 
0247:             df = reconstruct_df(
0248:                 alpha=a,
0249:                 mach=M,
0250:                 cr=cr,
0251:                 ci=ci,
0252:                 ln_p_start_right=ln_amp,
0253:                 max_y_limit=args.max_y_limit,
0254:                 mapping_scale=args.mapping_scale,
0255:                 max_step=ms,
0256:             )
0257: 
0258:             df["max_step_label"] = label
```

Lines 269–281

```python
0269:                 "Mach": M,
0270:                 "alpha": a,
0271:                 "cr": cr,
0272:                 "ci": ci,
0273:                 "max_step_label": label,
0274:                 "max_step": ms,
0275:                 "ln_p_start_right": ln_amp,
0276:                 "stage2_mismatch": stage2,
0277:                 "y_limit": y_limit,
0278:                 "fields_file": str(fpath),
0279:             })
0280: 
0281:             print(f"  max_step={label}: stage2={stage2:.2e}, y_limit={y_limit:.1f}", flush=True)
```

Lines 311–323

```python
0311:             comp_rows.append(row)
0312: 
0313:     comp = pd.DataFrame(comp_rows)
0314:     comp.to_csv(args.output_dir / "max_step_convergence_metrics.csv", index=False)
0315: 
0316:     # PDF overlay Re.
0317:     pdf_path = args.output_dir / "max_step_real_oscillation_overlay.pdf"
0318: 
0319:     with PdfPages(pdf_path) as pdf:
0320:         for _, r in cand.iterrows():
0321:             M = float(r["Mach"])
0322:             a = float(r["alpha"])
0323: 
```

### `code/scripts/audits/audit_M18_M19_refined_convergence.py`

Lines 14–29

```python
0014:     reconstruct_dataframe,
0015:     field_shape_metrics,
0016: )
0017: 
0018: 
0019: FIELD_SPECS = [
0020:     ("p", "p_real", "p_imag"),
0021:     ("rho", "rho_real", "rho_imag"),
0022:     ("u", "u_real", "u_imag"),
0023:     ("v", "v_real", "v_imag"),
0024: ]
0025: 
0026: 
0027: SETTINGS = [
0028:     # name, mapping_scale, search_y_limit, final_y_limit
0029:     ("ms1p5_y1200", 1.5, 900.0, 1200.0),
```

Lines 187–199

```python
0187: 
0188:     fields = reconstruct_dataframe(
0189:         alpha=alpha,
0190:         mach=mach,
0191:         cr=cr,
0192:         ci=ci,
0193:         ln_p_start_right=ln_amp,
0194:         max_y_limit=final_y_limit,
0195:         mapping_scale=mapping_scale,
0196:     )
0197: 
0198:     metrics = field_shape_metrics(fields)
0199: 
```

Lines 206–218

```python
0206:         "final_y_limit": final_y_limit,
0207:         "cr": cr,
0208:         "ci": ci,
0209:         "omega_i": alpha * ci,
0210:         "stage1_mismatch": stage1,
0211:         "stage2_mismatch_exact": stage2,
0212:         "ln_p_start_right_exact": ln_amp,
0213:         "actual_y_limit": y_limit,
0214:         "amplitude_status": amp_status,
0215:     }
0216:     row.update(metrics)
0217: 
0218:     return row, fields
```

### `code/scripts/audits/audit_supersonic_shooting_visual_validation.py`

Lines 33–52

```python
0033:     parser.add_argument("--output-stem", type=str, required=True)
0034:     return parser
0035: 
0036: 
0037: def infer_summary_columns(df: pd.DataFrame) -> tuple[str, str, str]:
0038:     triplets = [
0039:         ("best_shooting_cr", "best_shooting_ci", "best_ln_p_start_right"),
0040:         ("shooting_cr", "shooting_ci", "ln_p_start_right"),
0041:         ("cr", "ci", "ln_p_start_right"),
0042:     ]
0043:     for cr_col, ci_col, ln_col in triplets:
0044:         if {cr_col, ci_col, ln_col}.issubset(df.columns):
0045:             return cr_col, ci_col, ln_col
0046:     raise ValueError("Impossible d'identifier les colonnes cr/ci/ln_p_start_right dans le summary CSV.")
0047: 
0048: 
0049: def parse_bool_like(value: object, *, default: bool) -> bool:
0050:     if value is None or (isinstance(value, float) and np.isnan(value)):
0051:         return default
0052:     if isinstance(value, (bool, np.bool_)):
```

Lines 108–120

```python
0108: def reconstruct_shooting_fields(
0109:     *,
0110:     alpha: float,
0111:     mach: float,
0112:     cr: float,
0113:     ci: float,
0114:     ln_p_start_right: float,
0115:     match_y: float,
0116:     use_mapping: bool,
0117:     mapping_scale: float,
0118:     min_y_limit: float,
0119:     max_y_limit: float,
0120:     y_limit_factor: float,
```

Lines 128–140

```python
0128:         mapping_scale=mapping_scale,
0129:         min_y_limit=min_y_limit,
0130:         max_y_limit=max_y_limit,
0131:         y_limit_factor=y_limit_factor,
0132:         max_step=max_step,
0133:     )
0134:     sol_left, _, sol_right_full, _ = solver.get_trajectories(cr, ci, ln_p_start_right=ln_p_start_right)
0135:     if not (sol_left.success and sol_right_full.success):
0136:         raise RuntimeError(
0137:             f"Echec de reconstruction du mode shooting pour alpha={alpha:.3f}, M={mach:.3f}."
0138:         )
0139: 
0140:     y_left = np.asarray(sol_left.t)
```

Lines 156–169

```python
0156: 
0157:     left_mask = y_left < 0.0
0158:     y = np.concatenate([y_left[left_mask], y_right[::-1]])
0159:     p = np.concatenate([p_left[left_mask], p_right[::-1]])
0160:     gamma = np.concatenate([gamma_left[left_mask], gamma_right[::-1]])
0161: 
0162:     u_bar = solver.base_velocity(y)
0163:     du_bar = solver.base_velocity_derivative(y)
0164:     c = complex(cr, ci)
0165:     i_alpha = 1j * float(alpha)
0166: 
0167:     p_y = gamma * p
0168:     v = -p_y / (i_alpha * (u_bar - c))
0169:     u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
```

Lines 236–251

```python
0236:     output_path: Path,
0237:     threshold_ratio: float,
0238:     min_half_width: float,
0239: ) -> None:
0240:     field_names = ["rho", "u", "v", "p"]
0241:     field_titles = [
0242:         r"Density Perturbation $\hat{\rho}$",
0243:         r"Streamwise Velocity $\hat{u}$",
0244:         r"Vertical Velocity $\hat{v}$",
0245:         r"Pressure Perturbation $\hat{p}$",
0246:     ]
0247: 
0248:     with PdfPages(output_path) as pdf:
0249:         for page in field_pages:
0250:             row = page["summary_row"]
0251:             principal = page["principal"]
```

Lines 310–322

```python
0310:         raise ValueError(f"Summary CSV missing required columns: {missing}")
0311: 
0312:     summary_out = summary_df.rename(
0313:         columns={
0314:             cr_col: "shooting_cr",
0315:             ci_col: "shooting_ci",
0316:             ln_col: "ln_p_start_right",
0317:         }
0318:     ).copy()
0319: 
0320:     field_rows: list[dict[str, float | str]] = []
0321:     visual_summary_rows: list[dict[str, float]] = []
0322:     field_pages: list[dict[str, object]] = []
```

Lines 324–336

```python
0324:     for _, row in summary_out.sort_values(["alpha", "Mach"]).iterrows():
0325:         principal = reconstruct_shooting_fields(
0326:             alpha=float(row["alpha"]),
0327:             mach=float(row["Mach"]),
0328:             cr=float(row["shooting_cr"]),
0329:             ci=float(row["shooting_ci"]),
0330:             ln_p_start_right=float(row["ln_p_start_right"]),
0331:             match_y=float(row["match_y"]) if "match_y" in row.index and pd.notna(row["match_y"]) else 1.0,
0332:             use_mapping=parse_bool_like(row["use_mapping"], default=True) if "use_mapping" in row.index else True,
0333:             mapping_scale=float(row["mapping_scale"]) if "mapping_scale" in row.index and pd.notna(row["mapping_scale"]) else 5.0,
0334:             min_y_limit=float(row["min_y_limit"]) if "min_y_limit" in row.index and pd.notna(row["min_y_limit"]) else 10.0,
0335:             max_y_limit=float(row["max_y_limit"]) if "max_y_limit" in row.index and pd.notna(row["max_y_limit"]) else 80.0,
0336:             y_limit_factor=float(row["y_limit_factor"]) if "y_limit_factor" in row.index and pd.notna(row["y_limit_factor"]) else 4.0,
```

Lines 381–400

```python
0381:                 field_rows.append(
0382:                     {
0383:                         "alpha": float(row["alpha"]),
0384:                         "Mach": float(row["Mach"]),
0385:                         "mode_family": mode_name,
0386:                         "y": float(y_value),
0387:                         "rho_real": float(np.real(rho_value)),
0388:                         "rho_imag": float(np.imag(rho_value)),
0389:                         "u_real": float(np.real(u_value)),
0390:                         "u_imag": float(np.imag(u_value)),
0391:                         "v_real": float(np.real(v_value)),
0392:                         "v_imag": float(np.imag(v_value)),
0393:                         "p_real": float(np.real(p_value)),
0394:                         "p_imag": float(np.imag(p_value)),
0395:                     }
0396:                 )
0397: 
0398:     visual_summary_df = pd.DataFrame(visual_summary_rows)
0399:     fields_df = pd.DataFrame(field_rows)
0400: 
```

