"""Click CLI for operating FLASH simulations from agent workflows."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

import click

from cli_anything.flash import __version__
from cli_anything.flash.core import auth
from cli_anything.flash.core import doctor as doctor_core
from cli_anything.flash.core import export as export_core
from cli_anything.flash.core import freepath
from cli_anything.flash.core import resolution as resolution_core
from cli_anything.flash.core.project import (
    FlashProjectError,
    create_project,
    discover_projects,
    get_param,
    list_files,
    parse_flash_par,
    project_status,
    resolve_project,
    resolve_root,
    set_params,
)
from cli_anything.flash.core.session import Session
from cli_anything.flash.utils import accelerator, backend


def _require_auth(ctx: click.Context) -> None:
    """Require a valid password before touching server-backed resources."""

    if ctx.obj.get("auth_ok"):
        return
    password = ctx.obj.get("password")
    if password is None and not ctx.obj.get("json") and sys.stdin.isatty():
        password = click.prompt("FLASH skill password", hide_input=True, confirmation_prompt=False)
    auth.require_password(password)
    ctx.obj["auth_ok"] = True


def _emit(ctx: click.Context, payload: dict[str, Any]) -> None:
    payload = export_core.finite_json(payload)
    if ctx.obj.get("json"):
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if "message" in payload:
        click.echo(payload["message"])
    else:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _root(ctx: click.Context) -> Path:
    _require_auth(ctx)
    return resolve_root(ctx.obj.get("root"))


def _project(ctx: click.Context, project: str | None = None) -> Path:
    root = _root(ctx)
    value = project or ctx.obj.get("project")
    path = resolve_project(root, value)
    assert path is not None
    return path


def _parse_assignments(assignments: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in assignments:
        if "=" not in item:
            raise click.ClickException(f"Expected NAME=VALUE, got {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise click.ClickException(f"Empty parameter name in {item}")
        out[key] = value.strip()
    return out


def _read_stdin_or_file(text: str | None, file_path: str | None) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    if text:
        return text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise click.ClickException("Provide --text, --file, or pipe input on stdin.")


def handle_errors(func):
    """Decorator for consistent JSON and human-readable error handling."""

    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        try:
            return func(*args, **kwargs)
        except click.exceptions.Exit:
            raise
        except Exception as exc:  # click needs one top-level conversion point
            payload = {"ok": False, "error": str(exc), "type": exc.__class__.__name__}
            if ctx.obj and ctx.obj.get("json"):
                click.echo(json.dumps(payload, ensure_ascii=False, indent=2), err=False)
                raise click.exceptions.Exit(1)
            raise click.ClickException(str(exc)) from exc

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


@click.group(invoke_without_command=True)
@click.option("--root", envvar="FLASH_ROOT", help="FLASH root directory. Defaults to FLASH_ROOT or common local paths.")
@click.option("--project", envvar="FLASH_PROJECT", help="Default FLASH project/run directory name.")
@click.option("--password", envvar="FLASH_SKILL_PASSWORD", help="Password required before connecting to the FLASH server tree.")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--dry-run", is_flag=True, help="Validate a mutation without writing when supported.")
@click.version_option(__version__)
@click.pass_context
def cli(
    ctx: click.Context,
    root: str | None,
    project: str | None,
    password: str | None,
    json_output: bool,
    dry_run: bool,
) -> None:
    """Operate FLASH simulations through an agent-friendly CLI."""

    ctx.ensure_object(dict)
    ctx.obj.update(
        {
            "root": root,
            "project": project,
            "password": password,
            "json": json_output,
            "dry_run": dry_run,
            "auth_ok": False,
        }
    )
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl)


@cli.group()
def auth_cmd() -> None:
    """Check FLASH skill password access."""


cli.add_command(auth_cmd, name="auth")


@auth_cmd.command("check")
@click.pass_context
@handle_errors
def auth_check(ctx: click.Context) -> None:
    """Verify that the provided password can connect to the FLASH server tree."""

    _require_auth(ctx)
    root = resolve_root(ctx.obj.get("root"))
    _emit(ctx, {"ok": True, "authenticated": True, "root": str(root)})


@cli.command()
@click.pass_context
@handle_errors
def discover(ctx: click.Context) -> None:
    """List FLASH projects under the root."""

    root = _root(ctx)
    _emit(ctx, {"ok": True, "root": str(root), "projects": discover_projects(root)})


@cli.command()
@click.pass_context
@handle_errors
def status(ctx: click.Context) -> None:
    """Show root or project status."""

    root = _root(ctx)
    _emit(ctx, {"ok": True, **project_status(root, ctx.obj.get("project"))})


@cli.group()
def project() -> None:
    """Create and list FLASH projects."""


@project.command("list")
@click.pass_context
@handle_errors
def project_list(ctx: click.Context) -> None:
    """List available project directories."""

    root = _root(ctx)
    _emit(ctx, {"ok": True, "root": str(root), "projects": discover_projects(root)})


@project.command("create")
@click.argument("name")
@click.option("--template", required=True, help="Existing project name to copy.")
@click.option("--copy-mode", type=click.Choice(["light", "full"]), default="light", show_default=True)
@click.pass_context
@handle_errors
def project_create(ctx: click.Context, name: str, template: str, copy_mode: str) -> None:
    """Create a new project from an existing project."""

    if ctx.obj.get("dry_run"):
        _emit(ctx, {"ok": True, "dry_run": True, "would_create": name, "template": template, "copy_mode": copy_mode})
        return
    _emit(ctx, {"ok": True, **create_project(_root(ctx), name, template, copy_mode)})


@cli.group()
def param() -> None:
    """Read and edit flash.par parameters."""


@param.command("list")
@click.option("--search", default="", help="Filter by parameter name or raw value.")
@click.pass_context
@handle_errors
def param_list(ctx: click.Context, search: str) -> None:
    """List parameters from flash.par."""

    path = _project(ctx)
    rows = parse_flash_par(path / "flash.par")
    if search:
        low = search.lower()
        rows = [row for row in rows if low in row["name"].lower() or low in row["raw_value"].lower()]
    _emit(ctx, {"ok": True, "project": path.name, "count": len(rows), "parameters": rows})


@param.command("get")
@click.argument("name")
@click.pass_context
@handle_errors
def param_get(ctx: click.Context, name: str) -> None:
    """Get one flash.par parameter."""

    path = _project(ctx)
    _emit(ctx, {"ok": True, "project": path.name, "parameter": get_param(path, name)})


@param.command("set")
@click.argument("assignments", nargs=-1, required=True)
@click.option("--allow-new", is_flag=True, help="Append missing parameters.")
@click.pass_context
@handle_errors
def param_set(ctx: click.Context, assignments: tuple[str, ...], allow_new: bool) -> None:
    """Set one or more parameters: NAME=VALUE."""

    path = _project(ctx)
    updates = _parse_assignments(assignments)
    if ctx.obj.get("dry_run"):
        _emit(ctx, {"ok": True, "dry_run": True, "project": path.name, "updates": updates, "allow_new": allow_new})
        return
    _emit(ctx, {"ok": True, **set_params(path, updates, allow_new=allow_new)})


@cli.group()
def resolution() -> None:
    """Inspect and edit simulation mesh resolution."""


@resolution.command("show")
@click.pass_context
@handle_errors
def resolution_show(ctx: click.Context) -> None:
    """Show mesh size, domain spacing, and MPI layout from flash.par."""

    path = _project(ctx)
    _emit(ctx, {"ok": True, "resolution": resolution_core.summarize_resolution(path)})


@resolution.command("set")
@click.option("--i-grid-size", type=int, default=None, help="Set iGridSize, the X-direction mesh cells.")
@click.option("--j-grid-size", type=int, default=None, help="Set jGridSize, the Y-direction mesh cells.")
@click.option("--k-grid-size", type=int, default=None, help="Set kGridSize, the Z-direction mesh cells.")
@click.option("--i-procs", type=int, default=None, help="Set iProcs, the X-direction processor decomposition.")
@click.option("--j-procs", type=int, default=None, help="Set jProcs, the Y-direction processor decomposition.")
@click.option("--k-procs", type=int, default=None, help="Set kProcs, the Z-direction processor decomposition.")
@click.option("--mesh-copy-count", type=int, default=None, help="Set meshCopyCount used in MPI rank accounting.")
@click.option("--xmin", type=float, default=None, help="Set physical domain xmin.")
@click.option("--xmax", type=float, default=None, help="Set physical domain xmax.")
@click.option("--ymin", type=float, default=None, help="Set physical domain ymin.")
@click.option("--ymax", type=float, default=None, help="Set physical domain ymax.")
@click.option("--zmin", type=float, default=None, help="Set physical domain zmin.")
@click.option("--zmax", type=float, default=None, help="Set physical domain zmax.")
@click.option("--allow-new", is_flag=True, help="Append missing resolution parameters to flash.par.")
@click.pass_context
@handle_errors
def resolution_set(
    ctx: click.Context,
    i_grid_size: int | None,
    j_grid_size: int | None,
    k_grid_size: int | None,
    i_procs: int | None,
    j_procs: int | None,
    k_procs: int | None,
    mesh_copy_count: int | None,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    zmin: float | None,
    zmax: float | None,
    allow_new: bool,
) -> None:
    """Update mesh/domain resolution parameters in flash.par."""

    path = _project(ctx)
    updates = {
        "iGridSize": i_grid_size,
        "jGridSize": j_grid_size,
        "kGridSize": k_grid_size,
        "iProcs": i_procs,
        "jProcs": j_procs,
        "kProcs": k_procs,
        "meshCopyCount": mesh_copy_count,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }
    updates = {key: value for key, value in updates.items() if value is not None}
    if ctx.obj.get("dry_run"):
        _emit(
            ctx,
            {
                "ok": True,
                "dry_run": True,
                "project": path.name,
                "updates": updates,
                "allow_new": allow_new,
                "current": resolution_core.summarize_resolution(path),
            },
        )
        return
    _emit(ctx, {"ok": True, **resolution_core.update_resolution(path, updates, allow_new=allow_new)})


@cli.group()
def run() -> None:
    """Build, start, stop, and inspect FLASH runs."""


@run.command("start")
@click.argument("command", required=False, default="./flash4")
@click.pass_context
@handle_errors
def run_start(ctx: click.Context, command: str) -> None:
    """Start FLASH in the background."""

    path = _project(ctx)
    if ctx.obj.get("dry_run"):
        _emit(ctx, {"ok": True, "dry_run": True, "project": path.name, "command": command})
        return
    _emit(ctx, {"ok": True, **backend.start(path, command)})


@run.command("stop")
@click.pass_context
@handle_errors
def run_stop(ctx: click.Context) -> None:
    """Stop a background FLASH run started by this harness."""

    path = _project(ctx)
    _emit(ctx, {"ok": True, **backend.stop(path)})


@run.command("status")
@click.pass_context
@handle_errors
def run_status(ctx: click.Context) -> None:
    """Show run process status."""

    path = _project(ctx)
    _emit(ctx, {"ok": True, "project": path.name, "process": project_status(_root(ctx), path.name)["process"]})


@cli.command()
@click.option("-j", "--jobs", default=2, show_default=True, type=int)
@click.option("--timeout", default=3600.0, show_default=True, type=float)
@click.pass_context
@handle_errors
def build(ctx: click.Context, jobs: int, timeout: float) -> None:
    """Run make in the selected project."""

    path = _project(ctx)
    result = backend.build(path, jobs=jobs, timeout=timeout)
    _emit(ctx, {"ok": result["returncode"] == 0, **result})
    if result["returncode"] != 0:
        raise click.exceptions.Exit(result["returncode"])


@cli.group("accelerator")
def accelerator_cmd() -> None:
    """Build, inspect, and run selectable CPU/GPU FLASH backends."""


@accelerator_cmd.command("status")
@click.pass_context
@handle_errors
def accelerator_status(ctx: click.Context) -> None:
    """Show available backends and the recommended backend for flash.par."""

    root = _root(ctx)
    path = _project(ctx)
    _emit(ctx, {"ok": True, **accelerator.status(root, path)})


@accelerator_cmd.command("build-gpu")
@click.option("-j", "--jobs", default=2, show_default=True, type=int)
@click.option("--timeout", default=7200.0, show_default=True, type=float)
@click.pass_context
@handle_errors
def accelerator_build_gpu(ctx: click.Context, jobs: int, timeout: float) -> None:
    """Build flash4.gpu while preserving flash4.cpu."""

    root = _root(ctx)
    path = _project(ctx)
    if ctx.obj.get("dry_run"):
        _emit(
            ctx,
            {"ok": True, "dry_run": True, "project": path.name, "backend": "gpu", "jobs": jobs},
        )
        return
    _emit(ctx, {"ok": True, **accelerator.build_gpu(root, path, jobs, timeout)})


def _accelerator_options(function):
    function = click.option("--threads", type=int)(function)
    function = click.option("--ranks", type=int)(function)
    function = click.option(
        "--backend", "backend_name", type=click.Choice(("cpu", "gpu", "auto")), default="auto"
    )(function)
    return function


@accelerator_cmd.command("command")
@_accelerator_options
@click.pass_context
@handle_errors
def accelerator_command(
    ctx: click.Context, backend_name: str, ranks: int | None, threads: int | None
) -> None:
    """Print the exact MPI command selected for CPU, GPU, or auto mode."""

    root = _root(ctx)
    path = _project(ctx)
    spec = accelerator.launch_spec(root, path, backend_name, ranks, threads)
    _emit(ctx, {"ok": True, **spec, "shell_command": accelerator.shell_command(spec)})


@accelerator_cmd.command("run")
@_accelerator_options
@click.pass_context
@handle_errors
def accelerator_run(
    ctx: click.Context, backend_name: str, ranks: int | None, threads: int | None
) -> None:
    """Start the selected CPU/GPU backend in the background."""

    root = _root(ctx)
    path = _project(ctx)
    spec = accelerator.launch_spec(root, path, backend_name, ranks, threads)
    command = accelerator.shell_command(spec)
    if ctx.obj.get("dry_run"):
        _emit(ctx, {"ok": True, "dry_run": True, **spec, "shell_command": command})
        return
    _emit(ctx, {"ok": True, **spec, **backend.start(path, command)})


@cli.group()
def command() -> None:
    """Run native shell commands inside a FLASH project."""


@command.command("run")
@click.argument("shell_command")
@click.option("--timeout", default=60.0, show_default=True, type=float)
@click.pass_context
@handle_errors
def command_run(ctx: click.Context, shell_command: str, timeout: float) -> None:
    """Execute an arbitrary shell command in the project directory."""

    path = _project(ctx)
    result = backend.run_shell(path, shell_command, timeout=timeout)
    _emit(ctx, {"ok": result["returncode"] == 0, **result})
    if result["returncode"] != 0:
        raise click.exceptions.Exit(result["returncode"])


@cli.group()
def file() -> None:
    """List and inspect FLASH files."""


@file.command("list")
@click.option("--limit", default=200, show_default=True, type=int)
@click.pass_context
@handle_errors
def file_list(ctx: click.Context, limit: int) -> None:
    """List logs, parameters, tables, images, and HDF5 outputs."""

    path = _project(ctx)
    _emit(ctx, {"ok": True, "project": path.name, "files": list_files(path, limit=limit)})


@file.command("view")
@click.argument("name")
@click.option("--lines", default=120, show_default=True, type=int)
@click.pass_context
@handle_errors
def file_view(ctx: click.Context, name: str, lines: int) -> None:
    """View a text tail or summarize HDF5/table files."""

    path = _project(ctx)
    target = (path / name).resolve()
    target.relative_to(path.resolve())
    if not target.exists():
        raise FlashProjectError(f"File not found: {name}")
    if target.suffix.lower() in {".dat", ".csv"}:
        _emit(ctx, {"ok": True, "kind": "table", **export_core.table_summary(target)})
    elif export_core.is_hdf5(target):
        _emit(ctx, {"ok": True, "kind": "hdf5", **export_core.hdf5_summary(target)})
    else:
        _emit(ctx, {"ok": True, "kind": "text", **export_core.read_text_tail(target, lines=lines)})


@cli.group()
def plot() -> None:
    """Render FLASH outputs to PNG."""


@plot.command("hdf5")
@click.argument("name")
@click.option("--variable", required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--block", default=0, show_default=True, type=int)
@click.option("--z-index", default=0, show_default=True, type=int)
@click.option("--scale", type=click.Choice(["linear", "log10"]), default="linear", show_default=True)
@click.option("--width", default=7.0, show_default=True, type=float, help="Output figure width in inches.")
@click.option("--height", default=5.0, show_default=True, type=float, help="Output figure height in inches.")
@click.option("--dpi", default=140, show_default=True, type=int, help="Output image DPI.")
@click.pass_context
@handle_errors
def plot_hdf5(
    ctx: click.Context,
    name: str,
    variable: str,
    output: Path | None,
    block: int,
    z_index: int,
    scale: str,
    width: float,
    height: float,
    dpi: int,
) -> None:
    """Render one HDF5 dataset to PNG."""

    path = _project(ctx)
    target = (path / name).resolve()
    target.relative_to(path.resolve())
    if output is None:
        output = export_core.default_output_path(target)
    _emit(
        ctx,
        {
            "ok": True,
            **export_core.plot_hdf5(
                target,
                variable,
                output,
                block=block,
                z_index=z_index,
                scale=scale,
                width=width,
                height=height,
                dpi=dpi,
            ),
        },
    )


@plot.command("table")
@click.argument("name")
@click.option("--x-column", default=0, show_default=True, type=int)
@click.option("--y-column", default=1, show_default=True, type=int)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--scale", type=click.Choice(["linear", "log10"]), default="linear", show_default=True)
@click.option("--width", default=7.0, show_default=True, type=float, help="Output figure width in inches.")
@click.option("--height", default=5.0, show_default=True, type=float, help="Output figure height in inches.")
@click.option("--dpi", default=140, show_default=True, type=int, help="Output image DPI.")
@click.pass_context
@handle_errors
def plot_table(
    ctx: click.Context,
    name: str,
    x_column: int,
    y_column: int,
    output: Path | None,
    scale: str,
    width: float,
    height: float,
    dpi: int,
) -> None:
    """Render two numeric columns to PNG."""

    path = _project(ctx)
    target = (path / name).resolve()
    target.relative_to(path.resolve())
    if output is None:
        output = export_core.default_output_path(target)
    _emit(
        ctx,
        {
            "ok": True,
            **export_core.plot_table(
                target,
                output,
                x_column=x_column,
                y_column=y_column,
                scale=scale,
                width=width,
                height=height,
                dpi=dpi,
            ),
        },
    )


@cli.group()
def freepath_cmd() -> None:
    """Control and test the Rosseland free-path model."""


cli.add_command(freepath_cmd, name="freepath")


@freepath_cmd.command("control")
@click.option("--url", default=freepath.DEFAULT_URL, show_default=True)
@click.option("--enabled/--disabled", default=None, help="Update FLASH web control enable state.")
@click.option("--mode", type=click.Choice(["model", "hybrid", "truth"]), default=None)
@click.option("--elements", default=None, help="Comma/space list of Z values allowed to use model fallback.")
@click.pass_context
@handle_errors
def freepath_control(ctx: click.Context, url: str, enabled: bool | None, mode: str | None, elements: str | None) -> None:
    """Get or update the web model FLASH control endpoint."""

    _require_auth(ctx)
    update: dict[str, Any] = {}
    if enabled is not None:
        update["enabled"] = enabled
    if mode is not None:
        update["predictionMode"] = mode
    if elements is not None:
        update["modelElements"] = elements
    payload = freepath.control(url, update=update or None)
    _emit(ctx, {"ok": True, "control": payload})


@freepath_cmd.command("probe", context_settings={"allow_interspersed_args": False})
@click.option("--url", default=freepath.DEFAULT_URL, show_default=True)
@click.option("--mode", type=click.Choice(["model", "hybrid", "truth"]), default="hybrid", show_default=True)
@click.option("--elements", default=None)
@click.argument("z", type=float)
@click.argument("rod", type=float)
@click.argument("tep", type=float)
@click.argument("tgama", type=float)
@click.pass_context
@handle_errors
def freepath_probe(ctx: click.Context, url: str, mode: str, elements: str | None, z: float, rod: float, tep: float, tgama: float) -> None:
    """Predict one model-coordinate point: Z rod tep tgama."""

    _require_auth(ctx)
    _emit(ctx, {"ok": True, "prediction": freepath.probe(z, rod, tep, tgama, mode=mode, model_elements=elements, url=url)})


@freepath_cmd.command("batch")
@click.option("--url", default=freepath.DEFAULT_URL, show_default=True)
@click.option("--mode", type=click.Choice(["model", "hybrid", "truth"]), default="hybrid", show_default=True)
@click.option("--elements", default=None)
@click.option("--default-z", type=float, default=None)
@click.option("--text", default=None, help="Inline rows.")
@click.option("--file", "file_path", default=None, type=click.Path())
@click.pass_context
@handle_errors
def freepath_batch(ctx: click.Context, url: str, mode: str, elements: str | None, default_z: float | None, text: str | None, file_path: str | None) -> None:
    """Predict many rows. Format: Z rod tep tgama [true_lnu]."""

    _require_auth(ctx)
    rows_text = _read_stdin_or_file(text, file_path)
    _emit(ctx, {"ok": True, "prediction": freepath.batch(rows_text, mode=mode, model_elements=elements, default_z=default_z, url=url)})


def _clarify_payload(request: str, project: str | None, auto_modify: bool | None) -> dict[str, Any]:
    text = request.lower()
    run_words = ["run", "start", "simulate", "simulation", "运行", "模拟", "跑"]
    modify_words = ["set", "change", "修改", "设置", "调整", "改"]
    result_words = ["plot", "show result", "visual", "图", "结果", "查看", "画"]
    resolution_words = ["resolution", "mesh", "grid", "分辨率", "网格", "格点"]
    state_words = run_words + modify_words + ["new project", "新建", "rerun", "retry", "重跑", "再次模拟"]
    questions = []
    if project is None and any(word in text for word in run_words + ["stop", "build", "编译"] + modify_words + result_words):
        questions.append("要操作哪个 FLASH 项目目录？可以先运行 `cli-anything-flash --password <密码> --json project list`。")
    if any(word in text for word in run_words):
        if "flash4" not in text and "mpirun" not in text and "mpiexec" not in text:
            questions.append("启动命令是什么？例如 `./flash4`、`mpirun -np 4 ./flash4`，还是只做 dry-run/编译？")
        if not any(token in text for token in ["tmax", "nend", "checkpoint", "plotfile", "stop", "终止", "步数", "时间"]):
            questions.append("本次模拟的停止条件和输出频率是什么？例如 `tmax`、`nend`、plot/checkpoint 间隔。")
    if auto_modify is None and any(word in text for word in state_words):
        questions.append("是否允许自动修改参数并重跑？必须在开始时明确回答 yes/no；未明确时按 no 处理。")
    if any(word in text for word in modify_words) and "=" not in request:
        questions.append("要修改哪些 flash.par 参数，目标值分别是多少？格式建议为 `参数名=值`。")
    if any(word in text for word in resolution_words) and not any(
        token in text for token in ["igridsize", "jgridsize", "kgridsize", "i-grid", "j-grid", "k-grid", "64", "128", "256", "512"]
    ):
        questions.append(
            "仿真网格分辨率要设成多少？请给 `iGridSize/jGridSize/kGridSize`，以及对应的 `iProcs/jProcs/kProcs/meshCopyCount` 或 MPI 进程数。"
        )
    if any(word in text for word in ["freepath", "自由程", "模型"]) and not any(mode in text for mode in ["model", "hybrid", "truth", "只用模型", "混合", "真值"]):
        questions.append("自由程来源策略要用 model、hybrid 还是真值 truth？哪些 Z 元素允许模型兜底？")
    if any(word in text for word in result_words) and not any(s in text for s in ["hdf5", ".dat", ".csv", "trajectory", "dens", "tele", "tion", "magz"]):
        questions.append("要查看哪个结果文件和变量/列？HDF5 需要变量名，例如 dens、tele、tion、magz；表格需要 X/Y 列。")
    if any(word in text for word in ["new project", "新建"]) and "template" not in text and "模板" not in text:
        questions.append("新项目名字是什么？要从哪个已有项目作为模板复制？")
    if not questions:
        questions.append("信息基本足够；执行前仍应确认密码、项目名、自动修改开关和长时间运行风险。")
    return {
        "request": request,
        "questions": questions,
        "auto_modify": auto_modify,
        "requires_user_confirmation": auto_modify is None and any(word in text for word in state_words),
    }


@cli.command()
@click.option("--request", required=True, help="User's natural-language request.")
@click.option("--auto-modify/--no-auto-modify", default=None, help="Whether rerun loops may automatically edit parameters.")
@click.pass_context
@handle_errors
def clarify(ctx: click.Context, request: str, auto_modify: bool | None) -> None:
    """Return concise questions for an ambiguous FLASH request."""

    _emit(ctx, {"ok": True, **_clarify_payload(request, ctx.obj.get("project"), auto_modify)})


@cli.group()
def workflow() -> None:
    """Plan guarded simulation/review/rerun workflows."""


@workflow.command("plan")
@click.option("--request", required=True, help="User's natural-language request.")
@click.option("--auto-modify/--no-auto-modify", default=None, help="Whether automatic parameter edits and reruns are allowed.")
@click.pass_context
@handle_errors
def workflow_plan(ctx: click.Context, request: str, auto_modify: bool | None) -> None:
    """Return the safe workflow for a simulation request."""

    clarify_info = _clarify_payload(request, ctx.obj.get("project"), auto_modify)
    steps = [
        "Authenticate with `--password` or FLASH_SKILL_PASSWORD before connecting to the FLASH server root.",
        "Resolve missing parameters from the clarify questions before changing flash.par or starting a run.",
        "Run the simulation or requested command with JSON output and record the result.",
        "Run `cli-anything-flash --password <password> --json --project <project> doctor` to produce a structured error/result packet.",
        "Spawn a sub-agent with only the doctor packet, relevant log tails, command history, and requested objective; ask it to find errors and risky assumptions.",
        "If the sub-agent finds issues and auto_modify=true, edit the smallest necessary parameters, rerun, and repeat doctor/sub-agent review.",
        "If auto_modify=false or missing, ask the user before every parameter edit or rerun.",
        "Finish by interpreting the newest result artifacts and plots, then give next-step recommendations.",
    ]
    _emit(
        ctx,
        {
            "ok": True,
            **clarify_info,
            "workflow": steps,
            "subagent_review_prompt": (
                "Subagent task: review this FLASH run packet for errors, suspicious parameters, "
                "missing outputs, and whether the result interpretation is justified. "
                "Return findings, fixes, and whether rerun is needed."
            ),
        },
    )


@cli.command()
@click.option("--include-logs/--no-logs", default=True, show_default=True)
@click.pass_context
@handle_errors
def doctor(ctx: click.Context, include_logs: bool) -> None:
    """Inspect a project after a run and interpret available results."""

    path = _project(ctx)
    _emit(ctx, {"ok": True, "doctor": doctor_core.inspect_project(path, include_logs=include_logs)})


@cli.command()
@click.pass_context
@handle_errors
def repl(ctx: click.Context) -> None:
    """Open a small REPL that forwards commands to the CLI."""

    try:
        from cli_anything.flash.utils.repl_skin import ReplSkin
    except Exception:
        ReplSkin = None
    if ReplSkin:
        skin = ReplSkin("flash", version=__version__)
        skin.print_banner()
        skin.info("Type CLI subcommands such as `project list --json`; type `exit` to leave.")
    else:
        click.echo("cli-anything-flash REPL. Type `exit` to leave.")
    while True:
        try:
            line = input("flash> ").strip()
        except EOFError:
            break
        if line in {"exit", "quit"}:
            break
        if not line:
            continue
        args = shlex.split(line)
        try:
            cli.main(args=args, prog_name="cli-anything-flash", standalone_mode=False, obj=ctx.obj)
        except SystemExit as exc:
            if exc.code not in (0, None):
                click.echo(f"command exited: {exc.code}", err=True)
        except Exception as exc:
            click.echo(f"error: {exc}", err=True)


@cli.group()
def session() -> None:
    """Inspect undo/redo session metadata."""


@session.command("show")
@click.pass_context
@handle_errors
def session_show(ctx: click.Context) -> None:
    """Show session JSON."""

    sess = Session().load()
    _emit(ctx, {"ok": True, "path": str(sess.path), "session": sess.state})


@session.command("undo")
@click.pass_context
@handle_errors
def session_undo(ctx: click.Context) -> None:
    """Pop the last stored snapshot."""

    sess = Session().load()
    _emit(ctx, {"ok": True, "snapshot": sess.undo()})


@session.command("redo")
@click.pass_context
@handle_errors
def session_redo(ctx: click.Context) -> None:
    """Restore the last popped snapshot marker."""

    sess = Session().load()
    _emit(ctx, {"ok": True, "snapshot": sess.redo()})


def main() -> None:
    """Console entry point."""

    cli(obj={})


if __name__ == "__main__":
    main()
