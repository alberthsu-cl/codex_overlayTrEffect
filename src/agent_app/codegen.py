from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .artifacts import validate_effect_design
from .io import load_json, write_json


_DISSOLVE_ID = re.compile(r"^ModelGenerated\\Dissolve_(\d{2})$")
_MODEL_ID = re.compile(r"^ModelGenerated\\([A-Za-z][A-Za-z0-9]*)_(\d{2})$")
_TEMPLATE_FILES = (
    "TrGeneratedDissolve.h",
    "TrGeneratedDissolve.cpp",
    "TrGeneratedDissolve_ps.hlsl",
)


def generate_effect(
    design_file: Path,
    output_dir: Path,
    template_root: Path,
    manifest_file: Path,
    force: bool = False,
) -> dict[str, Any]:
    design = load_json(design_file)
    issues = validate_effect_design(design)
    if issues:
        raise ValueError("invalid effect design: " + "; ".join(issues))
    action = design["decision"]["action"]
    if action == "tune_existing_effect":
        return _generate_source_variant(
            design=design,
            design_file=design_file,
            template_root=template_root,
            output_dir=output_dir,
            manifest_file=manifest_file,
            force=force,
        )
    if action != "implement_new_effect":
        raise ValueError("code generation requires an implement_new_effect or tune_existing_effect decision")

    effect_id = design["target_effect"].get("effect_id")
    match = _DISSOLVE_ID.fullmatch(effect_id or "")
    if not match:
        raise ValueError(
            "the first generator supports effect IDs matching "
            "ModelGenerated\\Dissolve_XX"
        )

    symbol = f"ModelGeneratedDissolve{match.group(1)}"
    class_name = f"CTr{symbol}"
    shader_symbol = f"g_Tr_{symbol}_PS"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[str] = []
    for template_name in _TEMPLATE_FILES:
        template_path = template_root / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"effect template not found: {template_path}")

        output_name = template_name.replace("TrGeneratedDissolve", f"Tr{symbol}")
        output_path = output_dir / output_name
        if output_path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite generated file: {output_path}")

        source = template_path.read_text(encoding="utf-8")
        source = source.replace("CTrGeneratedDissolve", class_name)
        source = source.replace("TrGeneratedDissolve", f"Tr{symbol}")
        source = source.replace("g_Tr_GeneratedDissolve_PS", shader_symbol)
        output_path.write_text(source, encoding="utf-8")
        generated_files.append(str(output_path))

    manifest = {
        "manifest_type": "generated_effect",
        "manifest_version": 1,
        "effect_id": effect_id,
        "family": design["target_effect"].get("family"),
        "analysis_artifact": design.get("analysis_artifact"),
        "design_artifact": str(design_file.resolve()),
        "template": "dissolve",
        "class_name": class_name,
        "shader_symbol": shader_symbol,
        "generated_files": generated_files,
        "implementation_seed": design.get("implementation_seed"),
        "registration": {
            "fx_info_header": "overlaytrengine/OverlayTrPlugInFx/FxInfo.h",
            "plugin_source": "overlaytrengine/OverlayTrPlugInFx/OverlayTrPlugInFx.cpp",
            "project_file": "overlaytrengine/OverlayTrPlugInFx/OverlayTrPlugInFx.vcxproj",
            "required": True,
        },
    }
    write_json(manifest_file, manifest)
    return manifest


def _generate_source_variant(
    design: dict[str, Any],
    design_file: Path,
    template_root: Path,
    output_dir: Path,
    manifest_file: Path,
    force: bool,
) -> dict[str, Any]:
    variant = design.get("source_variant")
    target = design["target_effect"]
    effect_id = target.get("effect_id")
    if not isinstance(variant, dict):
        raise ValueError("tune_existing_effect requires a source_variant object")
    if not isinstance(effect_id, str) or not effect_id.startswith("ModelGenerated\\"):
        raise ValueError("source variants require a ModelGenerated\\... effect ID")

    base_stem = variant.get("base_stem")
    base_effect_id = target.get("base_effect_id") or target.get("closest_existing_effect_id")
    source_files = variant.get("source_files")
    replacements = variant.get("replacements", {})
    resource_files = variant.get("resource_files", [])
    resource_folder = variant.get("resource_folder")
    if not isinstance(base_stem, str) or not base_stem:
        raise ValueError("source_variant.base_stem is required")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("source_variant.source_files must be a non-empty array")
    if not isinstance(replacements, dict):
        raise ValueError("source_variant.replacements must be an object")
    if resource_files and (not isinstance(resource_folder, str) or not resource_folder):
        raise ValueError("source_variant.resource_folder is required when resources are declared")

    suffix = effect_id.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError("source variant effect IDs must end with a numeric index")
    symbol = "ModelGenerated" + effect_id.split("ModelGenerated", 1)[1].replace("\\", "").replace("_", "")
    class_name = f"CTr{symbol}"
    shader_symbol = f"g_Tr_{symbol}_PS"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[str] = []
    replacement_counts = {
        old: 0
        for old, new in replacements.items()
        if not _is_automatic_variant_replacement(
            old=old,
            new=new,
            base_stem=base_stem,
            base_effect_id=base_effect_id,
            effect_id=effect_id,
            symbol=symbol,
            class_name=class_name,
            shader_symbol=shader_symbol,
        )
    }
    for relative_name in source_files:
        if not isinstance(relative_name, str):
            raise ValueError("source_variant.source_files must contain strings")
        template_path = _resolve_variant_template_path(template_root, relative_name)
        if not template_path.exists():
            raise FileNotFoundError(f"source variant template not found: {template_path}")
        output_name = Path(relative_name).name.replace(base_stem, f"Tr{symbol}")
        output_path = output_dir / output_name
        if output_path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite generated file: {output_path}")
        source = template_path.read_text(encoding="utf-8")
        source = source.replace(f"C{base_stem}", class_name)
        source = source.replace(base_stem, f"Tr{symbol}")
        source = source.replace(f"g_Tr_{base_stem.removeprefix('Tr')}_PS", shader_symbol)
        for old, new in replacements.items():
            if not isinstance(old, str) or not isinstance(new, str):
                raise ValueError("source_variant.replacements must map strings to strings")
            if _is_automatic_variant_replacement(
                old=old,
                new=new,
                base_stem=base_stem,
                base_effect_id=base_effect_id,
                effect_id=effect_id,
                symbol=symbol,
                class_name=class_name,
                shader_symbol=shader_symbol,
            ):
                continue
            count = source.count(old)
            if count > 1:
                raise ValueError(f"source replacement must match exactly once: {old}")
            if count == 1:
                source = source.replace(old, new, 1)
                replacement_counts[old] += 1
        output_path.write_text(source, encoding="utf-8")
        generated_files.append(str(output_path))

    for old, count in replacement_counts.items():
        if count != 1:
            raise ValueError(f"source replacement must match exactly once: {old}")

    generated_resources: list[dict[str, str]] = []
    for relative_name in resource_files:
        if not isinstance(relative_name, str):
            raise ValueError("source_variant.resource_files must contain strings")
        source_path = template_root / relative_name
        if not source_path.exists():
            raise FileNotFoundError(f"source variant resource not found: {source_path}")
        resource_output = output_dir / "resources" / resource_folder / Path(relative_name).name
        if resource_output.exists() and not force:
            raise FileExistsError(f"refusing to overwrite generated resource: {resource_output}")
        resource_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, resource_output)
        generated_resources.append(
            {
                "source": str(resource_output),
                "runtime_relative_path": f"Resource/{resource_folder}/{resource_output.name}",
            }
        )

    manifest = {
        "manifest_type": "generated_effect",
        "manifest_version": 1,
        "effect_id": effect_id,
        "family": target.get("family"),
        "analysis_artifact": design.get("analysis_artifact"),
        "design_artifact": str(design_file.resolve()),
        "template": "source_variant",
        "base_effect_id": base_effect_id,
        "base_stem": base_stem,
        "class_name": class_name,
        "shader_symbol": shader_symbol,
        "generated_files": generated_files,
        "generated_resources": generated_resources,
        "source_replacements": replacements,
        "implementation_seed": design.get("implementation_seed"),
        "registration": {"required": True},
    }
    write_json(manifest_file, manifest)
    return manifest


def _resolve_variant_template_path(template_root: Path, source_name: str) -> Path:
    """Accept template-relative paths and the repo-relative form the agent may emit."""
    relative_path = Path(source_name)
    direct_path = template_root / relative_path
    if direct_path.exists():
        return direct_path

    parts = relative_path.parts
    for index, part in enumerate(parts):
        if part.lower() != template_root.name.lower():
            continue
        repo_relative_path = template_root.joinpath(*parts[index + 1 :])
        if repo_relative_path.exists():
            return repo_relative_path
    return direct_path


def _is_automatic_variant_replacement(
    old: str,
    new: str,
    base_stem: str,
    base_effect_id: Any,
    effect_id: str,
    symbol: str,
    class_name: str,
    shader_symbol: str,
) -> bool:
    """Ignore identity transformations already performed by source-variant generation."""
    automatic_replacements = {
        base_stem: f"Tr{symbol}",
        f"C{base_stem}": class_name,
        f"g_Tr_{base_stem.removeprefix('Tr')}_PS": shader_symbol,
    }
    if automatic_replacements.get(old) == new:
        return True
    return isinstance(base_effect_id, str) and old == base_effect_id and new == effect_id


def register_effect(manifest_file: Path, target_root: Path) -> dict[str, Any]:
    """Copy a generated package into OverlayTrPlugInFx and register it safely."""
    manifest = load_json(manifest_file)
    effect_id = manifest.get("effect_id")
    match = _MODEL_ID.fullmatch(effect_id or "")
    if not match:
        raise ValueError("registration supports effect IDs matching ModelGenerated\\Family_XX")

    target_dir = target_root / "OverlayTrPlugInFx"
    fx_info_path = target_dir / "FxInfo.h"
    plugin_path = target_dir / "OverlayTrPlugInFx.cpp"
    project_path = target_dir / "OverlayTrPlugInFx.vcxproj"
    filters_path = target_dir / "OverlayTrPlugInFx.vcxproj.filters"
    source_paths = [Path(path) for path in manifest.get("generated_files", [])]
    if len(source_paths) != len(_TEMPLATE_FILES) or any(not path.exists() for path in source_paths):
        raise ValueError("manifest does not contain all existing generated source files")

    if manifest.get("generated_resources"):
        raise ValueError("pure-HLSL registration does not accept generated resources yet")
    suffix = int(match.group(2))
    symbol = f"ModelGenerated{match.group(1)}{match.group(2)}"
    class_name = manifest.get("class_name") or f"CTr{symbol}"
    shader_symbol = manifest.get("shader_symbol") or f"g_Tr_{symbol}_PS"
    base_stem = manifest.get("base_stem") or "TrGeneratedDissolve"
    destination_paths = [target_dir / path.name for path in source_paths]
    raw_target_text = {
        fx_info_path: _read_text_preserving_newlines(fx_info_path),
        plugin_path: _read_text_preserving_newlines(plugin_path),
        project_path: _read_text_preserving_newlines(project_path),
    }
    if filters_path.exists():
        raw_target_text[filters_path] = _read_text_preserving_newlines(filters_path)
    target_text = {path: content.replace("\r\n", "\n") for path, content in raw_target_text.items()}
    index_values = [int(value) for value in re.findall(r"\n\s+(\d+)\n", target_text[fx_info_path])]
    index = max(index_values, default=0) + 1

    if any(path.exists() for path in destination_paths):
        raise FileExistsError("refusing to overwrite an existing target effect source")
    cpp_effect_id = effect_id.replace("\\", "\\\\")
    if cpp_effect_id in target_text[fx_info_path]:
        raise ValueError(f"effect ID is already registered: {effect_id}")
    if class_name in target_text[plugin_path] or shader_symbol in target_text[project_path]:
        raise ValueError("generated class or shader symbol is already present")

    include_anchor = f'#include "{base_stem}.h"'
    fx_info_anchor = "\t};\n}"
    switch_anchor = "\t\tdefault:\n"
    project_compile_anchor = f'    <ClCompile Include="{base_stem}.cpp" />'
    project_include_anchor = f'    <ClInclude Include="{base_stem}.h" />'
    project_shader_anchor = f'    <FxCompile Include="{base_stem}_ps.hlsl">'
    for path, anchor in (
        (fx_info_path, include_anchor),
        (fx_info_path, fx_info_anchor),
        (plugin_path, switch_anchor),
        (project_path, project_compile_anchor),
        (project_path, project_include_anchor),
        (project_path, project_shader_anchor),
    ):
        if anchor not in target_text[path]:
            raise ValueError(f"registration anchor not found in {path.name}: {anchor}")

    fx_info_entry = (
        "\t\t{\n"
        "\t\t\t// Agent-generated dissolve effect\n"
        f'\t\t\t"{cpp_effect_id}",\n'
        f'\t\t\t"Generated {match.group(1)} {match.group(2)}",\n'
        f"\t\t\t{index}\n"
        "\t\t}\n"
    )
    case = f"\t\tcase {index}:\n\t\t\tm_pFx = new {class_name}(g_hInst, pFxParam->wszReferencePath);\n\t\t\tbreak;\n"
    shader_entry = _shader_project_entry(f"Tr{symbol}_ps.hlsl", shader_symbol)

    updated = dict(target_text)
    fx_info_with_include = updated[fx_info_path].replace(
        include_anchor, f'{include_anchor}\n#include "Tr{symbol}.h"', 1
    )
    fx_info_prefix, fx_info_suffix = fx_info_with_include.split(fx_info_anchor, 1)
    if not fx_info_prefix.rstrip().endswith("},"):
        closing_brace = fx_info_prefix.rfind("}")
        if closing_brace < 0:
            raise ValueError("could not locate the final FX table entry")
        fx_info_prefix = (
            fx_info_prefix[:closing_brace]
            + "},"
            + fx_info_prefix[closing_brace + 1 :]
        )
    updated[fx_info_path] = fx_info_prefix + fx_info_entry + fx_info_anchor + fx_info_suffix
    updated[plugin_path] = updated[plugin_path].replace(switch_anchor, case + switch_anchor, 1)
    updated[project_path] = updated[project_path].replace(
        project_compile_anchor,
        f'{project_compile_anchor}\n    <ClCompile Include="Tr{symbol}.cpp" />',
        1,
    ).replace(
        project_include_anchor,
        f'{project_include_anchor}\n    <ClInclude Include="Tr{symbol}.h" />',
        1,
    ).replace(project_shader_anchor, shader_entry + "\n" + project_shader_anchor, 1)

    if filters_path in updated:
        updated[filters_path] = _update_project_filters(
            updated[filters_path],
            base_stem=base_stem,
            cpp_filename=f"Tr{symbol}.cpp",
            header_filename=f"Tr{symbol}.h",
            shader_filename=f"Tr{symbol}_ps.hlsl",
        )

    for source, destination in zip(source_paths, destination_paths):
        shutil.copyfile(source, destination)
    for path, content in updated.items():
        newline = "\r\n" if "\r\n" in raw_target_text[path] else "\n"
        path.write_text(content.replace("\n", newline), encoding="utf-8", newline="")

    registration = {
        "effect_id": effect_id,
        "index": index,
        "class_name": class_name,
        "shader_symbol": shader_symbol,
        "target_files": [str(path) for path in destination_paths],
        "registration_files": [str(fx_info_path), str(plugin_path), str(project_path)],
    }
    manifest["registration"] = registration
    write_json(manifest_file, manifest)
    return registration


def initialize_candidate(
    manifest_file: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Copy a registered effect into an isolated workspace for agent edits."""
    manifest = load_json(manifest_file)
    registration = manifest.get("registration")
    if not isinstance(registration, dict):
        raise ValueError("generated effect manifest has no registration data")
    target_files = registration.get("target_files")
    if not isinstance(target_files, list) or not target_files:
        raise ValueError("registered effect manifest has no target files")
    source_paths = [Path(path) for path in target_files]
    if any(not path.exists() for path in source_paths):
        raise FileNotFoundError("registered effect source file is missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_files: list[str] = []
    for source_path in source_paths:
        destination = output_dir / source_path.name
        if destination.exists() and not force:
            raise FileExistsError(f"refusing to overwrite candidate file: {destination}")
        shutil.copyfile(source_path, destination)
        candidate_files.append(str(destination))

    candidate_manifest = {
        "manifest_type": "effect_candidate",
        "manifest_version": 1,
        "effect_id": manifest.get("effect_id"),
        "family": manifest.get("family"),
        "iteration": 0,
        "source_manifest": str(manifest_file),
        "analysis_artifact": manifest.get("analysis_artifact"),
        "design_artifact": manifest.get("design_artifact"),
        "candidate_files": candidate_files,
        "target_files": [str(path) for path in source_paths],
        "status": "active",
    }
    manifest_path = output_dir / "candidate_manifest.json"
    write_json(manifest_path, candidate_manifest)
    candidate_manifest["manifest_file"] = str(manifest_path)
    return candidate_manifest


def promote_candidate(
    candidate_manifest_file: Path,
    backup_dir: Path,
) -> dict[str, Any]:
    """Back up registered files and promote an isolated candidate in place."""
    candidate = load_json(candidate_manifest_file)
    candidate_files = candidate.get("candidate_files")
    target_files = candidate.get("target_files")
    if not isinstance(candidate_files, list) or not isinstance(target_files, list):
        raise ValueError("candidate manifest must contain candidate_files and target_files")
    if len(candidate_files) != len(target_files) or not candidate_files:
        raise ValueError("candidate and target file lists must have equal non-zero length")
    source_paths = [Path(path) for path in candidate_files]
    target_paths = [Path(path) for path in target_files]
    if any(not path.exists() for path in source_paths):
        raise FileNotFoundError("candidate source file is missing")
    if any(not path.exists() for path in target_paths):
        raise FileNotFoundError("registered target file is missing")

    backup_dir.mkdir(parents=True, exist_ok=True)
    backups: list[str] = []
    for source_path, target_path in zip(source_paths, target_paths):
        backup_path = backup_dir / target_path.name
        if backup_path.exists():
            raise FileExistsError(f"refusing to overwrite backup file: {backup_path}")
        shutil.copyfile(target_path, backup_path)
        shutil.copyfile(source_path, target_path)
        backups.append(str(backup_path))

    result = {
        "status": "succeeded",
        "effect_id": candidate.get("effect_id"),
        "candidate_manifest": str(candidate_manifest_file),
        "promoted_files": [str(path) for path in target_paths],
        "backup_files": backups,
    }
    candidate["status"] = "promoted"
    candidate["last_promotion"] = result
    write_json(candidate_manifest_file, candidate)
    return result


def _update_project_filters(
    text: str,
    *,
    base_stem: str,
    cpp_filename: str,
    header_filename: str,
    shader_filename: str,
) -> str:
    """Add generated files to the stable Solution Explorer filter."""
    filter_name = "Transition\\ModelGenerated"
    if f'<Filter Include="{filter_name}">' not in text:
        filter_block = (
            f'    <Filter Include="{filter_name}">\n'
            "      <UniqueIdentifier>{b7cf5f0d-3d70-4b3c-8d4b-3d66d64a8421}</UniqueIdentifier>\n"
            "    </Filter>\n"
        )
        text = text.replace("  </ItemGroup>", filter_block + "  </ItemGroup>", 1)

    def add_after(anchors: tuple[str, ...], entry: str) -> None:
        nonlocal text
        if entry not in text:
            for anchor in anchors:
                if anchor in text:
                    text = text.replace(anchor, anchor + "\n" + entry, 1)
                    return
            raise ValueError(f"filter anchor not found: {anchors[0]}")

    add_after(
        (
            f'    <ClCompile Include="{base_stem}.cpp">\n'
            f'      <Filter>Transition\\{base_stem}</Filter>\n'
            "    </ClCompile>",
            f'    <ClCompile Include="{base_stem}.cpp">\n'
            f'      <Filter>{filter_name}</Filter>\n'
            "    </ClCompile>",
        ),
        f'    <ClCompile Include="{cpp_filename}">\n      <Filter>{filter_name}</Filter>\n    </ClCompile>',
    )
    add_after(
        (
            f'    <ClInclude Include="{base_stem}.h">\n'
            f'      <Filter>Transition\\{base_stem}</Filter>\n'
            "    </ClInclude>",
            f'    <ClInclude Include="{base_stem}.h">\n'
            f'      <Filter>{filter_name}</Filter>\n'
            "    </ClInclude>",
        ),
        f'    <ClInclude Include="{header_filename}">\n      <Filter>{filter_name}</Filter>\n    </ClInclude>',
    )
    add_after(
        (
            f'    <FxCompile Include="{base_stem}_ps.hlsl">\n'
            f'      <Filter>Transition\\{base_stem}</Filter>\n'
            "    </FxCompile>",
            f'    <FxCompile Include="{base_stem}_ps.hlsl">\n'
            f'      <Filter>{filter_name}</Filter>\n'
            "    </FxCompile>",
        ),
        f'    <FxCompile Include="{shader_filename}">\n      <Filter>{filter_name}</Filter>\n    </FxCompile>',
    )
    return text


def _shader_project_entry(filename: str, shader_symbol: str) -> str:
    return (
        f'    <FxCompile Include="{filename}">\n'
        '      <EntryPointName Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'">Pixel_Shader</EntryPointName>\n'
        '      <ShaderType Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'">Pixel</ShaderType>\n'
        '      <ShaderModel Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'">4.0</ShaderModel>\n'
        f'      <VariableName Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'">{shader_symbol}</VariableName>\n'
        '      <HeaderFileOutput Condition="\'$(Configuration)|$(Platform)\'==\'Debug|x64\'">$(ProjectDir)Shader\\%(Filename).h</HeaderFileOutput>\n'
        '      <EntryPointName Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'">Pixel_Shader</EntryPointName>\n'
        '      <ShaderType Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'">Pixel</ShaderType>\n'
        '      <ShaderModel Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'">4.0</ShaderModel>\n'
        f'      <VariableName Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'">{shader_symbol}</VariableName>\n'
        '      <HeaderFileOutput Condition="\'$(Configuration)|$(Platform)\'==\'Release|x64\'">$(ProjectDir)Shader\\%(Filename).h</HeaderFileOutput>\n'
        '    </FxCompile>'
    )


def _read_text_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()
