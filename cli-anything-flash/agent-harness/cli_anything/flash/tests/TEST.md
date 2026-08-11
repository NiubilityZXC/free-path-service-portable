# cli-anything-flash Test Plan and Results

## Part 1: Test Plan

### Test Inventory Plan

- `test_core.py`: 10 unit tests planned.
- `test_full_e2e.py`: 10 end-to-end/subprocess tests planned.

### Unit Test Plan

- `project.py`: root discovery, project listing, `flash.par` parsing, parameter updates, project creation validation.
- `export.py`: numeric table parsing, table plotting, HDF5 summaries when fixture files are available.
- `freepath.py`: payload shape through mocked request helper.
- `session.py`: undo/redo state transitions.
- `auth.py`: password hash accepts only the configured secret.
- `doctor.py`: log error detection, missing executable detection, interpretation, next-step recommendations.
- `resolution.py`: simulation mesh/domain resolution summary, validated updates, backup creation, invalid-domain rejection.

### E2E Test Plan

- Resolve the real FLASH root through `FLASH_ROOT` or local defaults.
- Invoke the installed command via `_resolve_cli("cli-anything-flash")`.
- Verify JSON output for `--help`, `discover`, `status`, `param list`, `command run`.
- Verify server-backed commands reject missing/wrong password and accept the correct password.
- Verify the major server command families reject missing password: project, params, build, run, command, file, plot, resolution, doctor, freepath.
- Create a temporary FLASH project from `ZPinch2D_UG`, edit simulation resolution and `flash.par`, view files, start/stop a harmless background command, render HDF5 and DAT outputs when sample files exist.
- Verify `workflow plan` asks for missing details and exposes sub-agent review guidance.
- Verify `doctor` returns issues, interpretation, and next-step recommendations.
- Verify the free-path service command family if `http://127.0.0.1:8790/health` is reachable.

### Realistic Workflow Scenarios

- **FLASH setup inspection**: discover projects, select `improved_MRT`, list opacity parameters.
- **Controlled parameter edit**: copy a project and change a harmless string parameter with automatic backup.
- **Run lifecycle**: authenticate, start `sleep 30` in the project, confirm status, stop it.
- **Result visualization**: render `dens` from a sample HDF5 plot file and plot `trajectory.dat`.
- **Image resolution control**: render HDF5 and table PNG files with custom `--width`, `--height`, and `--dpi`, then verify actual pixel dimensions.
- **Simulation resolution control**: read `iGridSize/jGridSize/kGridSize`, domain spacing, total cells, and MPI layout; update a temporary project to `64x64x1` with single-process decomposition and verify derived values.
- **Review loop**: run `doctor`, package issues/interpretation/next steps for a sub-agent review.
- **Free-path model control**: read/update web control and batch-predict model points when the service is running.

## Part 2: Test Results

Command:

```bash
PATH=/home/user/miniconda3/envs/ai/bin:$PATH \
CLI_ANYTHING_FORCE_INSTALLED=1 \
FLASH_ROOT=/home/user2/Haigerloch/FLASH4.8 \
FLASH_SKILL_PASSWORD=<provided by user at test time> \
/home/user/miniconda3/envs/ai/bin/python -m pytest -v --tb=short \
/home/user/cli-anything-flash/agent-harness/cli_anything/flash/tests
```

Result:

```text
collected 20 items

cli_anything/flash/tests/test_core.py::test_parse_flash_par PASSED
cli_anything/flash/tests/test_core.py::test_set_params_preserves_backup PASSED
cli_anything/flash/tests/test_core.py::test_create_project_light_ignores_outputs PASSED
cli_anything/flash/tests/test_core.py::test_resolution_summary_and_update PASSED
cli_anything/flash/tests/test_core.py::test_resolution_rejects_invalid_domain PASSED
cli_anything/flash/tests/test_core.py::test_table_summary_and_plot PASSED
cli_anything/flash/tests/test_core.py::test_session_undo_redo PASSED
cli_anything/flash/tests/test_core.py::test_password_hash_accepts_only_expected_secret PASSED
cli_anything/flash/tests/test_core.py::test_doctor_detects_log_errors_and_interprets PASSED
cli_anything/flash/tests/test_core.py::test_freepath_batch_payload_uses_web_endpoint PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_help PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_auth_rejects_missing_and_wrong_password PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_server_commands_reject_without_password PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_discover_and_status PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_param_list_and_get PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_command_run PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_temp_project_edit_run_stop_and_cleanup PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_plot_outputs_when_samples_exist PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_clarify PASSED
cli_anything/flash/tests/test_full_e2e.py::TestCLISubprocess::test_workflow_plan_and_doctor PASSED

20 passed in 17.43s
```

Coverage notes:

- Tested all harness command families through the installed CLI command.
- Tested password gating: missing and wrong passwords are rejected before connecting to the FLASH root; the correct password permits access.
- Tested stable root reporting at `/home/user2/Haigerloch/FLASH4.8`.
- Tested guarded workflow planning and `doctor` result/error interpretation.
- Tested output image resolution control: HDF5 `dens` rendered as 480x360 and `trajectory.dat` rendered as 500x400 in E2E tests.
- Tested simulation resolution control: `resolution show`, `resolution set`, derived total cells, MPI rank count, backup creation, and missing-password rejection.
- Real long FLASH physics runs are not launched by default; the lifecycle test uses `sleep 30` to avoid accidental production compute.
- A separate manual smoke test launches real `./flash4` in a temporary `improved_MRT` copy with `nend=0`, `tmax=0.0`, and single-process `iProcs=jProcs=kProcs=meshCopyCount=1`.
- Free-path service tests are performed separately when `http://127.0.0.1:8790` is reachable.

## Manual Simulation Resolution Verification

Verified against the real `improved_MRT` project:

```bash
cli-anything-flash --password '<provided by user>' --json --project improved_MRT resolution show
cli-anything-flash --password '<provided by user>' --json --dry-run --project improved_MRT resolution set --i-grid-size 256 --j-grid-size 256 --k-grid-size 1 --i-procs 1 --j-procs 1 --k-procs 1 --mesh-copy-count 1
```

Observed:

- Current mesh: `iGridSize=512`, `jGridSize=512`, `kGridSize=1`.
- Domain: `xmin=0.0`, `xmax=0.002`, `ymin=0.0`, `ymax=0.002`.
- Derived values: 2D mesh, `total_cells=262144`, `dx=3.90625e-06`, `dy=3.90625e-06`, `mpi_ranks=1`.
- Dry-run resolution update accepted and reported the proposed values without modifying `flash.par`.

## Manual Free-Path Service Verification

The local service was reachable at `http://127.0.0.1:8790/health`.

Verified commands:

```bash
cli-anything-flash --password '<provided by user>' --json freepath control --enabled --mode hybrid --elements 4,13,79
cli-anything-flash --password '<provided by user>' --json freepath probe --mode model --elements 13 13 -1.870 1.771 1.755
cli-anything-flash --password '<provided by user>' --json freepath probe --mode hybrid --elements 13 13 -1.870 1.771 1.755
cli-anything-flash --password '<provided by user>' --json freepath probe --mode truth 13 -1.870 1.771 1.755
printf '13 -1.870 1.771 1.755\n13 -1.869 1.771 1.755\n' | cli-anything-flash --password '<provided by user>' --json freepath batch --mode hybrid --elements 13
cli-anything-flash --password '<provided by user>' --json freepath control --disabled --mode hybrid --elements 4
```

Observed:

- `model` selected `model` and returned `lnu=0.028167483005471554`.
- `hybrid` selected `table_truth` for the exact Al sample and returned `lnu=0.0498257`.
- `truth` selected `table_truth` for the exact Al sample and returned `lnu=0.0498257`.
- Batch hybrid returned 2 rows with 1 table-truth row, 1 model row, and 0 errors.
- Control state was restored to disabled, hybrid, model elements `[4.0]`.

## Manual Real FLASH Smoke

A temporary project was created from `improved_MRT`, edited to run as a single process and stop immediately:

```bash
cli-anything-flash --password '<provided by user>' --json project create agent_flash_smoke_<timestamp> --template improved_MRT --copy-mode light
cli-anything-flash --password '<provided by user>' --json --project agent_flash_smoke_<timestamp> param set nend=0 tmax=0.0 iProcs=1 jProcs=1 kProcs=1 meshCopyCount=1 plotFileIntervalStep=0 checkpointFileIntervalStep=999999999 run_comment=agent_flash_smoke --allow-new
cli-anything-flash --password '<provided by user>' --json --project agent_flash_smoke_<timestamp> command run './flash4' --timeout 60
cli-anything-flash --password '<provided by user>' --json --project agent_flash_smoke_<timestamp> doctor --include-logs
```

Observed:

- `./flash4` return code: `0`.
- `doctor` issues: `0`.
- Missing runtime libraries: none.
- Artifacts detected: 1 log, 8 HDF5-like outputs, 5 numeric tables, 1 plot, 2 parameter files.
- Interpretation included HDF5 variables and recommended plotting `dens` plus table diagnostics.
- The temporary project was removed after the smoke test.
