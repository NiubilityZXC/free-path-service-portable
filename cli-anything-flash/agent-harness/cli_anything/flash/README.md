# cli-anything-flash

Agent-friendly CLI harness for the FLASH simulation tree.

## Install

```bash
cd /path/to/cli-anything-flash/agent-harness
python -m pip install -e .
export FLASH_ROOT=/path/to/FLASH4.8
```

Server-backed commands require the FLASH skill password:

```bash
cli-anything-flash --password '<password>' --json auth check
```

Server identity for remote use:

- Hostname: `user-Pro-ET900A-X9`
- Primary LAN IPv4: `192.168.110.64`
- ZeroTier IPv4: `10.202.51.181`
- SSH port: `22`
- SSH user: `Tsouzou_von_Habsburg-Hohenschaw` (account display name/comment: `user_flash`)

Use the LAN IP on the same physical network and the ZeroTier IP on the ZeroTier intranet. Do not store passwords in files.

## Basic Usage

```bash
cli-anything-flash --password '<password>' --json discover
cli-anything-flash --password '<password>' --project improved_MRT --json status
cli-anything-flash --password '<password>' --project improved_MRT --json param list --search opacity
cli-anything-flash --password '<password>' --project improved_MRT --json param set opacity_useFreePathModel=true
cli-anything-flash --password '<password>' --project improved_MRT --json resolution show
cli-anything-flash --password '<password>' --project improved_MRT --json resolution set --i-grid-size 256 --j-grid-size 256 --k-grid-size 1 --i-procs 1 --j-procs 1 --k-procs 1 --mesh-copy-count 1
cli-anything-flash --password '<password>' --project improved_MRT --json command run "tail -n 80 ZPinch_2D.log"
cli-anything-flash --password '<password>' --project improved_MRT --json plot hdf5 ZPinch_2D_forced_hdf5_plt_cnt_0000 --variable dens --width 4 --height 3 --dpi 120
cli-anything-flash --password '<password>' --project improved_MRT --json doctor
```

`resolution show` reports simulation mesh size, domain spacing, total cells, and the MPI rank count implied by `iProcs*jProcs*kProcs*meshCopyCount`. `resolution set` changes those `flash.par` values and makes a backup first.

## Free-Path Model

```bash
cli-anything-flash --password '<password>' --json freepath control
cli-anything-flash --password '<password>' --json freepath control --enabled --mode hybrid --elements 4,13,79
cli-anything-flash --password '<password>' --json freepath probe --mode hybrid 13 -1.870 1.771 1.755
printf "13 -1.870 1.771 1.755\n" | cli-anything-flash --password '<password>' --json freepath batch --mode hybrid
```

The service URL defaults to `http://127.0.0.1:8790`.

## Ambiguous Requests

```bash
cli-anything-flash --json clarify --request "帮我运行 FLASH 并看结果"
cli-anything-flash --json workflow plan --request "帮我运行 FLASH 并看结果" --no-auto-modify
```

Agents should ask the returned questions before mutating parameters, creating projects,
or launching long simulations.
