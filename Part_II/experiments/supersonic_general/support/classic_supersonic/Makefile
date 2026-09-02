PYTHON ?= python

.PHONY: verify release-verify spectral-tests asset-tests modal-sample-tests witness-tests witness-solver-smoke witness-modal-smoke calibrate-witness-spectral calibrate-witness-modal finalize-witness-modal prepare-modal build-witness solver-api inspect-witness-result modal-audit manifest checksums clean package-tests package-build article-assets

verify: spectral-tests asset-tests modal-sample-tests witness-tests package-tests manifest checksums

release-verify: verify witness-solver-smoke witness-modal-smoke

spectral-tests:
	$(PYTHON) -m pytest tests/regression/test_spectral_reference.py -q

asset-tests:
	$(PYTHON) -m pytest tests/regression/test_showcase_assets.py -q

modal-sample-tests:
	$(PYTHON) -m pytest tests/regression/test_modal_sample.py -q

witness-tests:
	$(PYTHON) -m pytest tests/regression/test_witness_case.py -q

witness-solver-smoke:
	$(PYTHON) scripts/validation/run_witness_spectral_reproduction.py

witness-modal-smoke:
	$(PYTHON) scripts/validation/calibrate_witness_modal_reproduction.py
	$(PYTHON) tools/check_witness_modal_reproduction.py

calibrate-witness-spectral:
	$(PYTHON) scripts/validation/calibrate_witness_spectral_reproduction.py

calibrate-witness-modal:
	$(PYTHON) scripts/validation/calibrate_witness_modal_reproduction.py

finalize-witness-modal:
	$(PYTHON) tools/finalize_witness_modal_calibration.py

prepare-modal:
	$(PYTHON) tools/prepare_modal_reference.py

build-witness:
	$(PYTHON) tools/create_witness_case.py

solver-api:
	$(PYTHON) tools/inspect_solver_api.py

inspect-witness-result:
	$(PYTHON) tools/inspect_witness_solve_result.py

modal-audit:
	$(PYTHON) tools/inspect_modal_reconstruction_path.py
	$(PYTHON) tools/probe_witness_trajectories.py

manifest:
	$(PYTHON) tools/build_release_manifest.py
	$(PYTHON) tools/update_manifest_with_witness.py

checksums:
	$(PYTHON) tools/build_checksums.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

package-tests:
	PYTHONPATH=src $(PYTHON) -m pytest tests/unit/test_package_installation.py -q

package-build:
	rm -rf build dist src/*.egg-info
	$(PYTHON) -m build

article-assets:
	$(PYTHON) scripts/plotting/build_article_assets.py
