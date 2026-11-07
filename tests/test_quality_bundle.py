import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_quality_workflows as builder
import install_workflows
import prepare_models


class QualityBundleTests(unittest.TestCase):
    def test_graph_links_and_downloaded_models(self):
        runtime = json.loads((ROOT / "config/scail2-runtime.json").read_text())
        assets = {item["target"].split("/", 1)[1] for item in runtime["comfyui"]["required_files"]}
        for path in (ROOT / "workflows").glob("*.json"):
            graph = json.loads(path.read_text(encoding="utf-8"))
            nodes = {node["id"]: node for node in graph["nodes"]}
            links = {link[0]: link for link in graph["links"]}
            self.assertEqual(len(nodes), len(graph["nodes"]), path.name)
            self.assertEqual(len(links), len(graph["links"]), path.name)
            for ident, origin, output, target, input_slot, kind in links.values():
                self.assertIn(ident, nodes[origin]["outputs"][output]["links"])
                self.assertEqual(nodes[target]["inputs"][input_slot]["link"], ident)
                self.assertIn(kind, nodes[target]["inputs"][input_slot]["type"].split(","))
                self.assertEqual(kind, nodes[origin]["outputs"][output]["type"])
            for node in nodes.values():
                for slot, inp in enumerate(node.get("inputs", [])):
                    if inp.get("link") is not None:
                        self.assertEqual(links[inp["link"]][3:5], [node["id"], slot])
                for slot, out in enumerate(node.get("outputs", [])):
                    for ident in out.get("links") or []:
                        self.assertEqual(links[ident][1:3], [node["id"], slot])
                if node["type"] in {"UNETLoader", "CLIPLoader", "VAELoader", "CLIPVisionLoader",
                                    "CheckpointLoaderSimple", "LoraLoaderModelOnly"}:
                    self.assertIn(node["widgets_values"][0], assets)

    def test_quality_controls_reach_generation(self):
        for multi in (False, True):
            graph = builder.build(multi)
            nodes = {node["id"]: node for node in graph["nodes"]}
            links = {link[0]: link for link in graph["links"]}
            self.assertEqual(nodes[96]["widgets_values"], [builder.DPO, 1.0])
            self.assertEqual(nodes[132]["widgets_values"][4:8], [40, 5.0, "uni_pc", "simple"])
            self.assertEqual(nodes[48]["widgets_values"], [3.0])
            # Trace the sampler's MODEL back through the shift and DPO to UNET.
            node = nodes[132]
            for expected in (48, 96, 37):
                inp = next(i for i in node["inputs"] if i["name"] == "model")
                node = nodes[links[inp["link"]][1]]
                self.assertEqual(node["id"], expected)
            self.assertTrue(nodes[6]["widgets_values"][0].strip())
            for input_name, expected in (("reference_image", 183 if multi else 103),
                                         ("reference_image_mask", 183 if multi else 107)):
                inp = next(i for i in nodes[132]["inputs"] if i["name"] == input_name)
                self.assertEqual(links[inp["link"]][1], expected)
            # Input resampling and output encoding share one FPS value.
            for ident, name in ((113, "force_rate"), (163, "frame_rate")):
                inp = next(i for i in nodes[ident]["inputs"] if i["name"] == name)
                self.assertEqual(links[inp["link"]][1], 157)

    def test_workflow_install_preserves_user_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, destination = root / "bundle", root / "user/workflows"
            source.mkdir()
            destination.mkdir(parents=True)
            (source / "quality.json").write_text("new")
            (destination / "quality.json").write_text("user edits")
            (destination / "my-own.json").write_text("untouched")
            install_workflows.install(source, destination)
            backups = list((root / "user/workflow-backups").rglob("quality.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), "user edits")
            self.assertEqual((destination / "quality.json").read_text(), "new")
            self.assertEqual((destination / "my-own.json").read_text(), "untouched")
            install_workflows.install(source, destination)
            self.assertEqual(len(list((root / "user/workflow-backups").rglob("quality.json"))), 1)

    def test_offline_mode_never_downloads_missing_lora(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"lightx2v_lora": {"lora_path": str(Path(folder) / "missing"), "min_bytes": 5}}
            with patch.dict(prepare_models.os.environ, {}, clear=True), \
                    patch.object(prepare_models, "snapshot_download") as download:
                with self.assertRaisesRegex(RuntimeError, "missing or too small"):
                    prepare_models.ensure_lightx2v_lora(config, skip_download=True)
                download.assert_not_called()

    def test_checksum_rejects_same_size_wrong_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights"
            path.write_bytes(b"good")
            digest = hashlib.sha256(b"good").hexdigest()
            prepare_models.require_file(path, 4, "test model", digest)
            path.write_bytes(b"evil")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                prepare_models.require_file(path, 4, "test model", digest)


if __name__ == "__main__":
    unittest.main()
