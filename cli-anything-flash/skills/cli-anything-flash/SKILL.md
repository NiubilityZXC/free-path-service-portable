---
name: cli-anything-flash
description: "Use when Codex needs to operate FLASH simulations from natural language through the CLI-Anything FLASH harness: discover projects, clarify ambiguous requests, edit flash.par, adjust simulation mesh resolution, create projects, build, run, stop, execute native FLASH shell commands, inspect logs/files, visualize HDF5 or table outputs, and control/test the Rosseland free-path model."
---

# CLI-Anything FLASH

Use `cli-anything-flash` for FLASH simulation work. Prefer JSON output for agent use.

## Language

优先使用中文回答用户，包括状态说明、错误解释、结果解读和下一步建议。只有在用户明确要求英文、需要保留命令/代码/日志原文、或引用 FLASH/CLI 原始英文输出时，才使用英文。

## Required Gate

Before connecting to the FLASH server tree, require the user to provide the FLASH skill password. Do not guess it, do not print it back, and do not proceed to server-backed commands until authentication succeeds.

Use one of these forms:

```bash
cli-anything-flash --password '<用户输入的密码>' --json auth check
FLASH_SKILL_PASSWORD='<用户输入的密码>' cli-anything-flash --json auth check
```

If authentication fails, ask the user to re-enter the password. Commands that touch the server root, project directories, free-path service, runs, files, or plots must include the password through `--password` or `FLASH_SKILL_PASSWORD`.

## Server Identity

Use these values when the skill must connect to the FLASH machine from another computer:

- Hostname: `user-Pro-ET900A-X9`
- Primary LAN IPv4: `192.168.110.64`
- ZeroTier IPv4: `10.202.51.181`
- SSH port: `22`
- SSH user: `Tsouzou_von_Habsburg-Hohenschaw` (account display name/comment: `user_flash`)

Prefer the primary LAN IP on the same physical network. Prefer the ZeroTier IP when the client is connected through the ZeroTier intranet or if the LAN DHCP address is unreachable. Never store or print the SSH/FLASH password; ask the user to enter it when needed.

The default server root is:

```bash
/home/user2/Haigerloch/FLASH4.8
```

Only use another root when the user explicitly supplies `--root` or `FLASH_ROOT`.

## Setup

If the command is not installed:

```bash
cd /home/user/cli-anything-flash/agent-harness
python -m pip install -e .
```

On another machine, copy this harness and set:

```bash
export FLASH_ROOT=/path/to/FLASH4.8
```

If the other machine cannot see the FLASH filesystem directly, SSH to the server first:

```bash
ssh Tsouzou_von_Habsburg-Hohenschaw@192.168.110.64
```

Then run `cli-anything-flash` on the server, or mount the server FLASH root locally and set `FLASH_ROOT` to that mounted path.

## Clarify First When Needed

If a user request is vague, run:

```bash
cli-anything-flash --json clarify --request "USER REQUEST"
```

Ask the returned questions before mutating `flash.par`, launching long runs, deleting files, or creating projects. Common missing details are project name, parameter/value, simulation mesh resolution, result file/variable, run command/MPI shape, stop condition/output frequency, free-path source mode, and whether automatic modification/rerun is allowed.

At the start of every simulation workflow, ask the user:

```text
是否允许自动修改参数并在发现错误后自动重跑？yes/no
```

If the answer is not explicit, treat it as `no`.

For a full guarded plan:

```bash
cli-anything-flash --json workflow plan --request "USER REQUEST" --no-auto-modify
cli-anything-flash --json workflow plan --request "USER REQUEST" --auto-modify
```

## Core Commands

```bash
cli-anything-flash --password '<密码>' --json discover
cli-anything-flash --password '<密码>' --project improved_MRT --json status
cli-anything-flash --password '<密码>' --project improved_MRT --json param list --search opacity
cli-anything-flash --password '<密码>' --project improved_MRT --json param get opacity_useFreePathModel
cli-anything-flash --password '<密码>' --project improved_MRT --json param set opacity_useFreePathModel=true
cli-anything-flash --password '<密码>' --project improved_MRT --json resolution show
cli-anything-flash --password '<密码>' --project MRT_test --json resolution set --i-grid-size 256 --j-grid-size 256 --k-grid-size 1 --i-procs 1 --j-procs 1 --k-procs 1 --mesh-copy-count 1
cli-anything-flash --password '<密码>' --json project create MRT_test --template improved_MRT --copy-mode light
cli-anything-flash --password '<密码>' --project MRT_test --json build -j 4
cli-anything-flash --password '<密码>' --project MRT_test --json run start "./flash4"
cli-anything-flash --password '<密码>' --project MRT_test --json run stop
cli-anything-flash --password '<密码>' --project MRT_test --json command run "tail -n 80 ZPinch_2D.log"
```

`command run` is the escape hatch for any native FLASH CLI operation not wrapped by a typed command.

## Simulation Resolution

Use `resolution show` before changing resolution. It reports `iGridSize/jGridSize/kGridSize`, domain bounds, derived `dx/dy/dz`, total cells, processor layout, expected MPI ranks, and a recommended run command.

Use `resolution set` to change simulation mesh resolution and processor decomposition. `iGridSize/jGridSize/kGridSize` control the number of computational cells. `iProcs*jProcs*kProcs*meshCopyCount` should match the MPI process count used to launch FLASH; for example, if the result says `mpi_ranks=4`, launch with `mpirun -np 4 ./flash4`. Increasing mesh resolution increases memory/runtime and can affect numerical stability, so confirm stop conditions before a long run.

## Results And Visualization

```bash
cli-anything-flash --password '<密码>' --project improved_MRT --json file list
cli-anything-flash --password '<密码>' --project improved_MRT --json file view flash.par
cli-anything-flash --password '<密码>' --project improved_MRT --json plot hdf5 ZPinch_2D_forced_hdf5_plt_cnt_0000 --variable dens --width 4 --height 3 --dpi 120
cli-anything-flash --password '<密码>' --project improved_MRT --json plot table trajectory.dat --x-column 0 --y-column 1 --width 5 --height 4 --dpi 100
cli-anything-flash --password '<密码>' --project improved_MRT --json doctor
```

Use the returned `output` PNG path for visual inspection. For image resolution, set `--width` and `--height` in inches plus `--dpi`; the JSON result returns `figure.pixels` so the agent can verify the actual PNG size. Use `doctor` after runs to collect logs, issues, artifact summaries, interpretation, and next-step recommendations.

## Review And Rerun Loop

After a run:

1. Run `doctor`.
2. Spawn a sub-agent when multi-agent tools are available. Give it only the `doctor` JSON, relevant command history, log tails, and the user objective.
3. Ask the sub-agent to check for errors, suspicious parameters, missing outputs, unjustified interpretation, and whether a rerun is needed.
4. If the sub-agent finds an issue and `auto_modify=true`, make the smallest parameter/code change needed, rerun, and repeat this loop.
5. If `auto_modify=false`, ask the user before changing parameters or rerunning.
6. Finish by interpreting the newest outputs and plots, then recommend the next simulation or diagnostic.

## Rosseland Free-Path Model

FLASH uses these `flash.par` parameters for the current model hook:

- `opacity_useFreePathModel`
- `opacity_freePathHelper`
- `opacity_freePathUrl`
- `opacity_freePathTimeout`

Web control and model probes:

```bash
cli-anything-flash --password '<密码>' --json freepath control
cli-anything-flash --password '<密码>' --json freepath control --enabled --mode hybrid --elements 4,13,79
cli-anything-flash --password '<密码>' --json freepath probe --mode hybrid 13 -1.870 1.771 1.755
printf "13 -1.870 1.771 1.755\n" | cli-anything-flash --password '<密码>' --json freepath batch --mode hybrid
```

Coordinate format is `Z rod tep tgama`, where `rod/tep/tgama` are log10 model coordinates. In FLASH, batch points should be collected per step/opactity call and sent to `/api/predict/batch`; avoid per-cell HTTP calls.

## Safety Rules

- Use `--dry-run` before project creation or parameter edits when uncertain.
- `param set` backs up `flash.par` automatically.
- Do not assume a long production run should start unless project, command, and runtime expectations are clear.
- If the free-path web service is not reachable, tell the user to start `/home/user/Rosseland_opa_new_hetero/equally/free_path_web_app.py` or pass `--url`.
- Never enable automatic parameter edits or reruns unless the user confirmed this at the start of the workflow.
