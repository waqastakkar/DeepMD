from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CLI_SOURCE = ROOT / "src" / "paddle" / "cli.py"
RUN_SCRIPT = ROOT / "run.sh"
TOPLEVEL_CLI = ROOT / "cli.py"
BEGIN_MARKER = "<!-- BEGIN GENERATED QUICKSTART -->"
END_MARKER = "<!-- END GENERATED QUICKSTART -->"


@dataclass
class CommandSpec:
    name: str
    help: str | None = None
    required_flags: list[str] = field(default_factory=list)


def _is_const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keyword_bool(node: ast.keyword, name: str) -> bool | None:
    if node.arg != name:
        return None
    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
        return node.value.value
    return None


def _extract_cli_commands(source: str) -> list[CommandSpec]:
    tree = ast.parse(source)
    commands: list[CommandSpec] = []

    class CommandVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.current: CommandSpec | None = None

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_parser":
                if node.args:
                    cmd_name = _is_const_str(node.args[0])
                    if cmd_name:
                        help_text = None
                        for kw in node.keywords:
                            if kw.arg == "help":
                                help_text = _is_const_str(kw.value)
                        self.current = CommandSpec(name=cmd_name, help=help_text)
                        commands.append(self.current)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and self.current is not None
            ):
                if node.args:
                    flag = _is_const_str(node.args[0])
                    if flag and flag.startswith("--"):
                        required = False
                        for kw in node.keywords:
                            value = _keyword_bool(kw, "required")
                            if value is True:
                                required = True
                        if required:
                            self.current.required_flags.append(flag)
            self.generic_visit(node)

    visitor = CommandVisitor()
    build_parser_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_parser"),
        None,
    )
    if build_parser_node:
        visitor.visit(build_parser_node)
    else:
        visitor.visit(tree)
    return commands


def _find_section(lines: list[str]) -> tuple[int, int]:
    try:
        start = lines.index(f"{BEGIN_MARKER}\n")
        end = lines.index(f"{END_MARKER}\n")
    except ValueError as exc:
        raise RuntimeError("Quickstart markers not found in README.md") from exc
    if end <= start:
        raise RuntimeError("Quickstart markers are out of order.")
    return start, end


def _detect_console_scripts(pyproject: Path) -> dict[str, str]:
    if not pyproject.exists():
        return {}
    scripts: dict[str, str] = {}
    in_section = False
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == "[project.scripts]"
            continue
        if not in_section or "=" not in line:
            continue
        name, value = [part.strip() for part in line.split("=", 1)]
        scripts[name] = value.strip('"').strip("'")
    return scripts


def _script_has_main(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "__name__" in text and "main(" in text


def _detect_entrypoint() -> tuple[str, str]:
    if RUN_SCRIPT.exists():
        return "run.sh", "./run.sh"

    scripts = _detect_console_scripts(ROOT / "pyproject.toml")
    if scripts:
        script_name = sorted(scripts.keys())[0]
        return "console_script", script_name

    if _script_has_main(TOPLEVEL_CLI):
        return "python_script", f"python {TOPLEVEL_CLI.name}"

    module_path = ROOT / "src" / "paddle" / "cli.py"
    if _script_has_main(module_path):
        return "python_module", "python -m paddle.cli"

    raise RuntimeError("No runnable CLI entry point detected.")


def _ensure_entrypoint_exists(entrypoint: str) -> str:
    if entrypoint.startswith("./"):
        if not (ROOT / entrypoint[2:]).exists():
            raise RuntimeError(f"Entry point {entrypoint} does not exist.")
        return entrypoint
    if entrypoint.startswith("python "):
        parts = entrypoint.split()
        if len(parts) >= 2 and parts[1].endswith(".py"):
            path = ROOT / parts[1]
            if not path.exists():
                raise RuntimeError(f"Entry point {entrypoint} does not exist.")
        return entrypoint
    if entrypoint.startswith("python -m "):
        module_name = entrypoint.split("python -m ", 1)[1].strip()
        module_path = ROOT / "src" / Path(*module_name.split("."))
        module_file = module_path.with_suffix(".py")
        if not module_file.exists():
            fallback = _detect_entrypoint()
            return fallback[1]
        return entrypoint
    if (ROOT / entrypoint).exists():
        return entrypoint
    raise RuntimeError(f"Entry point {entrypoint} does not exist.")


def _format_required_flags(flags: Iterable[str], values: dict[str, str]) -> str:
    rendered = []
    for flag in flags:
        value = values.get(flag, "<value>")
        rendered.append(f"{flag} {value}")
    return " ".join(rendered)


def _render_quickstart(commands: list[CommandSpec]) -> str:
    entry_type, entrypoint = _detect_entrypoint()
    entrypoint = _ensure_entrypoint_exists(entrypoint)
    entry_label = {
        "run.sh": "run via `./run.sh`.",
        "console_script": f"run via `{entrypoint}`.",
        "python_script": f"run via `{entrypoint}`.",
        "python_module": f"run via `{entrypoint}`.",
    }[entry_type]

    cmd_lookup = {cmd.name: cmd for cmd in commands}
    defaults = {
        "--config": "config.yaml",
        "--out": "outdir",
        "--prep": "out_prep/prep",
        "--data": "out_data/windows.npz",
        "--splits": "out_data/splits.json",
    }
    pipeline_flags = _format_required_flags(
        cmd_lookup.get("pipeline", CommandSpec("pipeline")).required_flags,
        defaults,
    )
    make_configs_flags = _format_required_flags(
        cmd_lookup.get("make_configs", CommandSpec("make_configs")).required_flags,
        defaults,
    )

    lines = [
        "Install the package with your preferred environment manager. Ensure that Python can import",
        "the package and that all simulation dependencies are available.",
        "",
        f"CLI entry point: {entry_label}",
        "",
        "Minimal pipeline invocation (full workflow):",
        "",
        "```bash",
        f"{entrypoint} pipeline {pipeline_flags} --out outdir".replace("  ", " ").strip(),
        "```",
        "",
        "Generate example configuration YAMLs (writes explicit/implicit configs):",
        "",
        "```bash",
        f"{entrypoint} make_configs {make_configs_flags} --out configs".replace("  ", " ").strip(),
        "```",
        "",
        "Example config files produced:",
        "",
        "- `configs/config-explicit-5ns.yaml`",
        "- `configs/config-implicit-5ns.yaml`",
        "",
        "Create a working config by copying one of the generated files (for example,",
        "`configs/config-explicit-5ns.yaml`) to `config.yml` and editing paths, GPU settings,",
        "and run lengths as needed before invoking the pipeline.",
        "",
        "### CLI commands",
        "",
    ]

    for cmd in sorted(commands, key=lambda c: c.name):
        if cmd.help:
            lines.append(f"- `{cmd.name}` — {cmd.help}")
        else:
            lines.append(f"- `{cmd.name}`")

    lines.extend(
        [
            "",
            "### Helpful CLI examples",
            "",
            "Run just the CMD stage:",
            "",
            "```bash",
            f"{entrypoint} cmd --config config.yml --out out_cmd",
            "```",
            "",
            "Run equilibration-prep and then build a training dataset:",
            "",
            "```bash",
            f"{entrypoint} prep --config config.yml --out out_prep",
            f"{entrypoint} data --prep out_prep/prep --out out_data",
            "```",
            "",
            "Train an ensemble model from a prepared dataset:",
            "",
            "```bash",
            f"{entrypoint} train --data out_data/windows.npz --splits out_data/splits.json --out out_models",
            "```",
            "",
            "Run equilibration + production only:",
            "",
            "```bash",
            f"{entrypoint} equil_prod --config config.yml --out out_prod",
            "```",
            "",
            "Generate alanine dipeptide benchmarks:",
            "",
            "```bash",
            f"{entrypoint} bench_alanine --out benchmarks/alanine",
            "```",
        ]
    )
    return "\n".join(lines)


def _update_readme() -> None:
    source = CLI_SOURCE.read_text(encoding="utf-8")
    commands = _extract_cli_commands(source)
    if not commands:
        raise RuntimeError("No CLI commands detected in src/paddle/cli.py")
    quickstart = _render_quickstart(commands)
    readme_lines = README.read_text(encoding="utf-8").splitlines(keepends=True)
    start, end = _find_section(readme_lines)
    new_block = [f"{BEGIN_MARKER}\n", *[f"{line}\n" for line in quickstart.splitlines()], f"{END_MARKER}\n"]
    updated = readme_lines[:start] + new_block + readme_lines[end + 1:]
    README.write_text("".join(updated), encoding="utf-8")


def main() -> int:
    _update_readme()
    print("README.md updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
