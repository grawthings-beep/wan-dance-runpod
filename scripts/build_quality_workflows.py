#!/usr/bin/env python3
"""Generate quality workflows while retaining the original fast workflow."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "workflows/scail2-comfyui-infinity-runpod.json"
DPO = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
PROMPT = (
    "A single character performs a continuous full-body dance in a stable scene. "
    "The character's distinctive facial proportions, eye shape, hairstyle, hair color, "
    "body proportions, costume colors, fabric patterns and accessories remain consistent "
    "throughout the shot. The arms, legs, torso and head move together with the dance, "
    "while hair and clothing move naturally. The character remains recognizable after "
    "turning or being briefly occluded. The background stays coherent."
)


def build(multi=False):
    graph = json.loads(FAST.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in graph["nodes"]}
    suffix = "quality-multiref" if multi else "quality"
    graph["id"] = "b7633254-67ba-4db9-811e-" + ("7293e05c0602" if multi else "7293e05c0601")
    graph["revision"] = 0
    by_id[96]["widgets_values"] = [DPO, 1.0]
    by_id[96]["title"] = "DPO detail LoRA — strength 1 / compare with 0"
    by_id[48]["widgets_values"] = [3.0]
    by_id[6]["widgets_values"] = [PROMPT]
    by_id[6]["title"] = "Describe YOUR character, costume and scene here"
    by_id[6]["size"] = [450, 260]
    by_id[6]["pos"] = [770, -1000]
    by_id[7]["widgets_values"] = [
        "identity changes, different face, changing hairstyle, altered costume, missing accessories, "
        "changing body proportions, deformed hands, extra fingers, extra limbs, flicker, blurry details, "
        "text, subtitles, watermark, static pose"
    ]
    sampler = by_id[132]["widgets_values"]
    sampler[4:8] = [40, 5.0, "uni_pc", "simple"]
    sampler[11] = 121  # five seconds at 24fps, including one window boundary
    sampler[12] = True
    by_id[132]["title"] = "QUALITY — 40 steps / max_frames 121 (0 = full video)"
    by_id[113]["widgets_values"]["frame_load_cap"] = 121
    by_id[113]["title"] = "Dance video — first 121 frames (set 0 for full video)"
    by_id[163]["widgets_values"]["filename_prefix"] = "SCAIL/scail2-" + suffix
    by_id[163]["widgets_values"]["trim_to_audio"] = True
    by_id[107]["widgets_values"] = ["0", "area", False]
    by_id[107]["title"] = "One character — largest tracked person (object 0)"
    by_id[109]["title"] = "SAM: describe the dancer in the video"
    by_id[115]["title"] = "SAM: describe the character in the reference"

    def new_node(node_id, kind, pos, inputs, outputs, widgets=None, title=None, size=None):
        node = {"id": node_id, "type": kind, "pos": pos, "size": size or [310, 150],
                "flags": {}, "order": len(graph["nodes"]), "mode": 0,
                "inputs": inputs, "outputs": outputs,
                "properties": {"Node name for S&R": kind}, "widgets_values": widgets or []}
        if title:
            node["title"] = title
        graph["nodes"].append(node)
        by_id[node_id] = node
        return node

    def port(name, kind):
        return {"name": name, "type": kind}

    def connect(origin, output, target, input_slot, kind):
        link_id = max((l[0] for l in graph["links"]), default=0) + 1
        graph["links"].append([link_id, origin, output, target, input_slot, kind])
        return link_id

    note = (
        "SCAIL-2 QUALITY / 単一キャラのダンス\n"
        "1. reference.png と driving.mp4 を選ぶ。プロンプトには実際の髪型・目・衣装・背景を追記。\n"
        "   例: A woman with silver twin tails and violet eyes wears a navy jacket with gold trim...\n"
        "   例文の特徴はそのまま使わず、自分の画像の特徴に置き換える。\n"
        "2. マスクプレビューを確認。青色が目的の人物を覆うこと。SAMの human は必要に応じて変更。\n"
        "3. DPO強度1と0を同じseedで比較。40 steps / CFG5 / shift3 / UniPC。LightX2Vなし。\n"
        "4. 初期値は24fps・121枚。長尺はLoad Videoのframe_load_capとInfinityのmax_framesを両方0へ。\n"
        "5. 解像度は参照画像を0.9MPにリサイズ後32の倍数へ。Infinityの幅/高さは接続値で上書き。\n"
        "品質向上・速度・必要VRAMは実GPU未測定。まず単一参照で比較。"
    )
    new_node(170, "Note", [-2150, -1350], [], [], [note], size=[940, 390])
    new_node(171, "PreviewImage", [-350, 460], [port("images", "IMAGE")], [], title="Driving mask: blue dancer / black background", size=[300, 260])
    new_node(172, "PreviewImage", [0, 460], [port("images", "IMAGE")], [], title="Reference mask: blue character / white background", size=[300, 260])
    connect(107, 0, 171, 0, "IMAGE")
    connect(107, 1, 172, 0, "IMAGE")

    if multi:
        extra = deepcopy(by_id[58])
        extra.update(id=180, pos=[-1570, 520], title="EXTRA VIEW — same character, same costume", size=[380, 500])
        # Reusing reference.png makes the graph loadable; replace with a real extra view.
        extra["outputs"] = [port("IMAGE", "IMAGE"), port("MASK", "MASK")]
        graph["nodes"].append(extra)
        by_id[180] = extra
        track = deepcopy(by_id[116])
        track.update(id=181, pos=[-1000, 600], title="SAM: extra reference view")
        graph["nodes"].append(track)
        by_id[181] = track
        colored = deepcopy(by_id[107])
        colored.update(id=182, pos=[-650, 600], title="Extra view: blue character / BLACK background")
        colored["inputs"] = colored["inputs"][:2]
        colored["widgets_values"] = ["0", "area", True]
        graph["nodes"].append(colored)
        by_id[182] = colored
        new_node(183, "WanSCAILReferenceBatch", [380, 400],
                 [port("reference_images", "IMAGE"), port("reference_masks", "IMAGE"),
                  port("extra_image", "IMAGE"), port("extra_mask", "IMAGE")],
                 [port("reference_images", "IMAGE"), port("reference_masks", "IMAGE")],
                 title="Same character: primary + fitted extra view")
        new_node(184, "PreviewImage", [380, 700], [port("images", "IMAGE")], [],
                 title="Both reference masks — primary white / extra black", size=[420, 300])
        graph["links"] = [l for l in graph["links"] if l[0] not in (267, 269)]
        for args in [(180, 0, 181, 0, "IMAGE"), (110, 0, 181, 1, "MODEL"),
                     (115, 0, 181, 3, "CONDITIONING"), (112, 0, 182, 0, "SAM3_TRACK_DATA"),
                     (181, 0, 182, 1, "SAM3_TRACK_DATA"), (103, 0, 183, 0, "IMAGE"),
                     (107, 1, 183, 1, "IMAGE"), (180, 0, 183, 2, "IMAGE"),
                     (182, 1, 183, 3, "IMAGE"), (183, 0, 132, 6, "IMAGE"),
                     (183, 1, 132, 7, "IMAGE"), (183, 1, 184, 0, "IMAGE")]:
            connect(*args)
        by_id[170]["widgets_values"][0] += (
            "\n追加参照は実験的。同じキャラ・同じ服装の顔アップ/背面等をEXTRA VIEWへ。"
            "初期のreference.png複製は別画像へ変更する。別キャラは入れない。"
            "追加画像は縦横比を保って収め、マスクも同じ変換をする。"
            "情報が増えても画質低下や二重化があり得るため、単一参照版と比較する。"
        )

    # Rebuild both ends of every link; cloned nodes must not retain stale link IDs.
    for order, node in enumerate(graph["nodes"]):
        node["order"] = order
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
    for link_id, origin, output, target, input_slot, kind in graph["links"]:
        by_id[origin]["outputs"][output]["links"].append(link_id)
        by_id[target]["inputs"][input_slot]["link"] = link_id
    graph["last_node_id"] = max(by_id)
    graph["last_link_id"] = max(l[0] for l in graph["links"])
    graph["groups"] = []
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for multi in (False, True):
        name = "scail2-quality" + ("-multiref" if multi else "") + "-runpod.json"
        path = ROOT / "workflows" / name
        content = json.dumps(build(multi), ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Regenerate {name}: python scripts/build_quality_workflows.py")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
        print(f"{'Checked' if args.check else 'Wrote'} {name}")


if __name__ == "__main__":
    main()
