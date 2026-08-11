# FLASH CLI-Anything Harness

## Architecture Analysis

FLASH 4.8 is primarily controlled through shell commands and project directories.
This harness uses the real FLASH tree instead of reimplementing the solver.

- Backend engine: compiled FLASH project directories containing `flash4`, `flash.par`,
  `Makefile`, generated Fortran/C objects, logs, HDF5 plot/checkpoint files, and text
  diagnostics.
- Data model: the project directory is the state boundary. Runtime configuration lives
  in `flash.par`; simulation output lives in HDF5, DAT, CSV, TXT, and log files.
- Existing CLI tools: `./setup`, `make`, `./flash4`, `mpirun`, ordinary shell tools,
  and the local Rosseland free-path web model at `http://127.0.0.1:8790`.
- Command bridge: high-frequency or low-level FLASH operations remain available through
  `command run`; common workflows are wrapped as typed commands.

## Command Map

- Project management: `project list`, `project create`, `status`
- Parameter control: `param list`, `param get`, `param set`
- Build/run: `build`, `run start`, `run stop`, `run status`
- Native shell access: `command run`
- Result inspection: `file list`, `file view`, `plot hdf5`, `plot table`
- Free-path model: `freepath control`, `freepath probe`, `freepath batch`
- Ambiguity handling: `clarify`

## Free-Path Integration

The current FLASH patch reads these `flash.par` parameters:

- `opacity_useFreePathModel`
- `opacity_freePathHelper`
- `opacity_freePathUrl`
- `opacity_freePathTimeout`

When the web control endpoint is enabled, FLASH should collect opacity points in
batches and call `/api/predict/batch`. It should not call HTTP for every cell and
group one by one. The CLI can toggle the parameters and can also test the web model
directly through `freepath probe` and `freepath batch`.

## Rendering Gap Assessment

FLASH has no built-in GUI in this local installation. The harness provides inspection
and plotting for files that are useful to agents:

- HDF5 plot/checkpoint fields are rendered to PNG through `h5py` and `matplotlib`.
- DAT/CSV/TXT numeric tables are plotted to PNG.
- Logs and text files can be tailed or summarized.
- Arbitrary external visualization tools can still be invoked with `command run`.
