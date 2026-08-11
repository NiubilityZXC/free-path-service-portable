---
name: cli-anything-flash
description: "Use when Codex needs to operate FLASH simulations from natural language through the CLI-Anything FLASH harness: discover projects, clarify ambiguous requests, edit flash.par, adjust simulation mesh resolution, create projects, build, run, stop, execute native FLASH shell commands, inspect logs/files, visualize HDF5 or table outputs, and control/test the Rosseland free-path model."
---

# CLI-Anything FLASH

Use `cli-anything-flash --json` for machine-readable FLASH operations. Before any command connects to `/home/user2/Haigerloch/FLASH4.8`, require the user to provide the FLASH skill password through `--password '<密码>'` or `FLASH_SKILL_PASSWORD`. Do not print the password back.

优先使用中文回答用户，包括状态说明、错误解释、结果解读和下一步建议。只有在用户明确要求英文、需要保留命令/代码/日志原文、或引用 FLASH/CLI 原始英文输出时，才使用英文。

Server identity for remote use:

- Hostname: `user-Pro-ET900A-X9`
- Primary LAN IPv4: `192.168.110.64`
- ZeroTier IPv4: `10.202.51.181`
- SSH port: `22`
- SSH user: `Tsouzou_von_Habsburg-Hohenschaw` (account display name/comment: `user_flash`)

Prefer the LAN IP on the same physical network and the ZeroTier IP over the ZeroTier intranet. Never store or print passwords; ask the user to enter them at connection/authentication time. If a different computer cannot see `/home/user2/Haigerloch/FLASH4.8` directly, SSH to the server or mount the server root before running `cli-anything-flash`.

At the start of each simulation workflow, ask whether automatic parameter edits and reruns are allowed. If the user does not explicitly say yes, treat auto modification as disabled.

Key commands:

```bash
cli-anything-flash --password '<密码>' --json auth check
cli-anything-flash --password '<密码>' --json discover
cli-anything-flash --password '<密码>' --project improved_MRT --json status
cli-anything-flash --password '<密码>' --project improved_MRT --json param list --search opacity
cli-anything-flash --password '<密码>' --project improved_MRT --json param set opacity_useFreePathModel=true
cli-anything-flash --password '<密码>' --project improved_MRT --json resolution show
cli-anything-flash --password '<密码>' --project improved_MRT --json resolution set --i-grid-size 256 --j-grid-size 256 --k-grid-size 1 --i-procs 1 --j-procs 1 --k-procs 1 --mesh-copy-count 1
cli-anything-flash --password '<密码>' --json project create MRT_test --template improved_MRT --copy-mode light
cli-anything-flash --password '<密码>' --project MRT_test --json run start "./flash4"
cli-anything-flash --password '<密码>' --project MRT_test --json run stop
cli-anything-flash --password '<密码>' --project MRT_test --json command run "tail -n 80 ZPinch_2D.log"
cli-anything-flash --password '<密码>' --project improved_MRT --json plot hdf5 ZPinch_2D_forced_hdf5_plt_cnt_0000 --variable dens --width 4 --height 3 --dpi 120
cli-anything-flash --password '<密码>' --project improved_MRT --json doctor
cli-anything-flash --password '<密码>' --json freepath control --enabled --mode hybrid --elements 4,13,79
```

When the user is unclear, run:

```bash
cli-anything-flash --json clarify --request "USER REQUEST"
cli-anything-flash --json workflow plan --request "USER REQUEST" --no-auto-modify
```

Ask the returned questions before changing parameters, launching long runs, changing simulation mesh resolution, or choosing a result file/variable. For simulation resolution, use `resolution show` and `resolution set`; `iGridSize/jGridSize/kGridSize` control computational cells, while `iProcs*jProcs*kProcs*meshCopyCount` must match the MPI rank count. For output image resolution, set `--width`, `--height`, and `--dpi`, then verify `figure.pixels` in JSON. After a run, execute `doctor`, delegate its JSON packet to a sub-agent for independent error review when multi-agent tools are available, and only auto-edit/rerun when the user confirmed auto modification at workflow start.

The validated unified candidate is `/home/user2/Haigerloch/FLASH4.8/GPU_hydro_lab_20260715/flash4.gpu-unified-v10`; consult `GPU_FUSED_NOTES.md`. It combines CUDA-HYPRE radiation/thermal/magnetic diffusion with the mixed gamma-law/IONMIX EOS offload and binds each node-local MPI rank to `local_rank modulo visible_device_count`. For PARAMESH, the launcher defaults to one rank per visible GPU, so all eligible GPUs are selected automatically. It estimates active work from the initial mesh and current checkpoint/log, then warns when the mesh is too small to keep the selected devices busy. It sets `FLASH_HYPRE_EXECUTION=device` and `FLASH_GPU_EOS=fused`; `FLASH_GPU_HLLD=off` remains the default because the selectable rank-HLLD experiment was slightly slower in the full run. Unsplit reconstruction/update, PARAMESH communication, source orchestration, and HDF5 I/O remain CPU work. `flash4.gpu-fused-v8` is the EOS-only fallback. Never use the numerically invalid v6 persistent-map experiment.
