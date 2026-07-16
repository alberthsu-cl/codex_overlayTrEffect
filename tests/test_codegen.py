from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import uuid
import unittest


AGENT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from agent_app.codegen import generate_effect, initialize_candidate, promote_candidate, register_effect


class CodegenTests(unittest.TestCase):
    def test_generate_dissolve_package(self) -> None:
        design = {
            "artifact_type": "effect_design",
            "artifact_version": 1,
            "analysis_artifact": "analysis.json",
            "decision": {"action": "implement_new_effect", "confidence": 0.8},
            "target_effect": {
                "family": "seamless",
                "effect_id": "ModelGenerated\\Dissolve_02",
            },
            "design_notes": {"must_preserve": [], "approximations": [], "risks": []},
        }

        root = Path(__file__).resolve().parents[1] / "work" / f"codegen_test_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            template_root = root / "templates"
            output_dir = root / "generated"
            template_root.mkdir()
            for name, content in {
                "TrGeneratedDissolve.h": "class CTrGeneratedDissolve {};",
                "TrGeneratedDissolve.cpp": '#include "TrGeneratedDissolve_ps.h"; void* x=g_Tr_GeneratedDissolve_PS;',
                "TrGeneratedDissolve_ps.hlsl": "float4 g_Tr_GeneratedDissolve_PS;",
            }.items():
                (template_root / name).write_text(content, encoding="utf-8")
            design_file = root / "design.json"
            design_file.write_text(json.dumps(design), encoding="utf-8")

            manifest = generate_effect(
                design_file,
                output_dir,
                template_root,
                root / "manifest.json",
            )

            self.assertEqual(manifest["class_name"], "CTrModelGeneratedDissolve02")
            self.assertTrue((output_dir / "TrModelGeneratedDissolve02.h").exists())
            self.assertIn("g_Tr_ModelGeneratedDissolve02_PS", (output_dir / "TrModelGeneratedDissolve02.cpp").read_text())
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_register_dissolve_package(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"register_test_{uuid.uuid4().hex}"
        plugin_dir = root / "OverlayTrPlugInFx"
        plugin_dir.mkdir(parents=True)
        try:
            (plugin_dir / "FxInfo.h").write_text(
                '#include "TrGeneratedDissolve.h"\nnamespace PlugInFxInfo {\n\tstatic FxInfo g_FxInfoList[] =\n\t{\n\t\t{\n\t\t\t"Existing",\n\t\t\t"Existing",\n\t\t\t22\n\t\t}\n\t};\n}\n',
                encoding="utf-8",
            )
            (plugin_dir / "OverlayTrPlugInFx.cpp").write_text("\t\tdefault:\n", encoding="utf-8")
            (plugin_dir / "OverlayTrPlugInFx.vcxproj").write_text(
                '    <ClCompile Include="TrGeneratedDissolve.cpp" />\n'
                '    <ClInclude Include="TrGeneratedDissolve.h" />\n'
                '    <FxCompile Include="TrGeneratedDissolve_ps.hlsl">\n'
                '    </FxCompile>\n',
                encoding="utf-8",
            )
            generated = root / "generated"
            generated.mkdir()
            generated_files = []
            for filename in ("TrModelGeneratedDissolve02.h", "TrModelGeneratedDissolve02.cpp", "TrModelGeneratedDissolve02_ps.hlsl"):
                path = generated / filename
                path.write_text(filename, encoding="utf-8")
                generated_files.append(str(path))
            manifest_file = root / "manifest.json"
            manifest_file.write_text(
                json.dumps({"effect_id": "ModelGenerated\\Dissolve_02", "generated_files": generated_files}),
                encoding="utf-8",
            )

            registration = register_effect(manifest_file, root)

            self.assertEqual(registration["index"], 23)
            self.assertIn("ModelGenerated\\\\Dissolve_02", (plugin_dir / "FxInfo.h").read_text())
            self.assertIn("CTrModelGeneratedDissolve02", (plugin_dir / "OverlayTrPlugInFx.cpp").read_text())
            self.assertTrue((plugin_dir / "TrModelGeneratedDissolve02.cpp").exists())
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_generate_source_variant_with_resources(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"variant_test_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            template_root = root / "templates"
            template_root.mkdir()
            (template_root / "TrSeamlessSliding.h").write_text(
                "class CTrSeamlessSliding {};", encoding="utf-8"
            )
            (template_root / "TrSeamlessSliding.cpp").write_text(
                '#include "TrSeamlessSliding_ps.h";\nfloat k = 1.2f;', encoding="utf-8"
            )
            (template_root / "TrSeamlessSliding_ps.hlsl").write_text(
                "float4 g_Tr_SeamlessSliding_PS;", encoding="utf-8"
            )
            (template_root / "variant.xml").write_text("<effect />", encoding="utf-8")
            design = {
                "artifact_type": "effect_design",
                "artifact_version": 1,
                "analysis_artifact": "analysis.json",
                "decision": {"action": "tune_existing_effect", "confidence": 0.8},
                "target_effect": {
                    "family": "seamless",
                    "effect_id": "ModelGenerated\\SeamlessSliding_01",
                    "base_effect_id": "CES_PlugIn_Seamless.dll\\DSP_TR_SeamlessSliding_LC",
                },
                "source_variant": {
                    "base_stem": "TrSeamlessSliding",
                    "source_files": [
                        "TrSeamlessSliding.h",
                        "TrSeamlessSliding.cpp",
                        "TrSeamlessSliding_ps.hlsl",
                    ],
                    "replacements": {"float k = 1.2f;": "float k = 0.8f;"},
                    "resource_folder": "ModelGenerated Seamless Transition",
                    "resource_files": ["variant.xml"],
                },
                "design_notes": {"must_preserve": [], "approximations": [], "risks": []},
            }
            design_file = root / "design.json"
            design_file.write_text(json.dumps(design), encoding="utf-8")

            manifest = generate_effect(
                design_file,
                root / "generated",
                template_root,
                root / "manifest.json",
            )

            self.assertEqual(manifest["template"], "source_variant")
            generated_cpp = root / "generated" / "TrModelGeneratedSeamlessSliding01.cpp"
            self.assertIn("0.8f", generated_cpp.read_text())
            self.assertIn("g_Tr_ModelGeneratedSeamlessSliding01_PS", (root / "generated" / "TrModelGeneratedSeamlessSliding01_ps.hlsl").read_text())
            self.assertEqual(len(manifest["generated_resources"]), 1)
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_candidate_init_and_promote(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"candidate_test_{uuid.uuid4().hex}"
        target_dir = root / "OverlayTrPlugInFx"
        target_dir.mkdir(parents=True)
        try:
            target_files = []
            for filename in ("TrModelGeneratedSeamlessSliding02.h", "TrModelGeneratedSeamlessSliding02.cpp"):
                path = target_dir / filename
                path.write_text(f"registered {filename}", encoding="utf-8")
                target_files.append(str(path))
            manifest_file = root / "generated_manifest.json"
            manifest_file.write_text(
                json.dumps({
                    "effect_id": "ModelGenerated\\SeamlessSliding_02",
                    "family": "seamless_slide",
                    "registration": {"target_files": target_files},
                }),
                encoding="utf-8",
            )

            candidate_dir = root / "candidate"
            initialized = initialize_candidate(manifest_file, candidate_dir)
            candidate_source = candidate_dir / "TrModelGeneratedSeamlessSliding02.cpp"
            candidate_source.write_text("refined candidate", encoding="utf-8")
            backup_dir = root / "backup"
            promoted = promote_candidate(candidate_dir / "candidate_manifest.json", backup_dir)

            self.assertEqual(initialized["effect_id"], "ModelGenerated\\SeamlessSliding_02")
            self.assertEqual(promoted["status"], "succeeded")
            self.assertEqual((target_dir / "TrModelGeneratedSeamlessSliding02.cpp").read_text(), "refined candidate")
            self.assertEqual(
                (backup_dir / "TrModelGeneratedSeamlessSliding02.cpp").read_text(),
                "registered TrModelGeneratedSeamlessSliding02.cpp",
            )
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()
