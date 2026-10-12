"""Build ComfyUI v1.0 and API workflows from the node's current input schemas."""

import ast
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "example_workflows"


def node_classes():
    tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
    selected = []
    constants = {"CATEGORY", "DEFAULT_CONFIG", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"}
    for statement in tree.body:
        if isinstance(statement, ast.ClassDef):
            selected.append(statement)
        elif isinstance(statement, ast.Assign) and any(isinstance(name, ast.Name) and name.id in constants for name in statement.targets):
            selected.append(statement)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "nodes.py schemas", "exec"), namespace)
    return namespace["NODE_CLASS_MAPPINGS"], namespace["NODE_DISPLAY_NAME_MAPPINGS"]


CLASSES, DISPLAY_NAMES = node_classes()
PRIMITIVES = {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}


class Workflow:
    def __init__(self, name):
        self.name, self.nodes, self.links, self.api, self.groups = name, [], [], {}, []

    def add(self, node_type, position, size, values=None, title=None):
        values = values or {}
        if node_type == "SaveAudioAdvanced":
            inputs = {"required": {"audio": ("AUDIO",), "filename_prefix": ("STRING", {"default": "audio/YuE2"}), "format": ("COMFY_DYNAMICCOMBO_V3", {"default": "flac"})}}
            outputs, names = ("AUDIO",), ("audio",)
        elif node_type == "LoadAudio":
            inputs = {"required": {"audio": (["select-source-audio.wav"],)}}
            outputs, names = ("AUDIO",), ("AUDIO",)
        else:
            node = CLASSES[node_type]
            inputs = node.INPUT_TYPES()
            outputs = node.RETURN_TYPES
            names = getattr(node, "RETURN_NAMES", outputs)
        index = len(self.nodes) + 1
        ports, widgets, named_widgets, api_inputs = [], [], {}, {}
        for section in ("required", "optional"):
            for name, definition in inputs.get(section, {}).items():
                input_type = "COMBO" if isinstance(definition[0], list) else definition[0]
                settings = definition[1] if len(definition) > 1 else {}
                port = {"name": name, "type": input_type, "link": None}
                if section == "optional":
                    port["shape"] = 7
                if input_type in PRIMITIVES and not settings.get("forceInput"):
                    port["widget"] = {"name": name}
                    default = definition[0][0] if isinstance(definition[0], list) else settings.get("default")
                    value = values.get(name, default)
                    widgets.append(value)
                    named_widgets[name] = value
                    api_inputs[name] = value
                    if name in {"seed", "noise_seed"} and settings.get("control_after_generate", True):
                        widgets.append("fixed")
                        named_widgets["control_after_generate"] = "fixed"
                ports.append(port)
        node = {
            "id": index, "type": node_type, "pos": list(position), "size": list(size),
            "flags": {}, "order": index - 1, "mode": 0, "inputs": ports,
            "outputs": [{"name": name, "type": value, "links": [], "slot_index": i} for i, (name, value) in enumerate(zip(names, outputs))],
            "title": title or DISPLAY_NAMES.get(node_type, node_type),
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets, "widgets_values_named": named_widgets,
            "color": "#24313c", "bgcolor": "#17222b",
        }
        node["properties"].update({"cnr_id": "comfy-core"} if node_type in {"SaveAudioAdvanced", "LoadAudio"} else {"aux_id": "jeankassio/ComfyUI_YuE2", "ver": "0.1.0"})
        self.nodes.append(node)
        self.api[str(index)] = {"class_type": node_type, "inputs": api_inputs, "_meta": {"title": node["title"]}}
        return index

    def connect(self, source, output_slot, destination, name):
        origin, target = self.nodes[source - 1], self.nodes[destination - 1]
        target_slot = next(i for i, port in enumerate(target["inputs"]) if port["name"] == name)
        output, port = origin["outputs"][output_slot], target["inputs"][target_slot]
        if output["type"] != port["type"]:
            raise ValueError(f"Incompatible connection: {origin['type']} -> {target['type']}.{name}")
        index = len(self.links) + 1
        self.links.append({"id": index, "origin_id": source, "origin_slot": output_slot, "target_id": destination, "target_slot": target_slot, "type": output["type"]})
        output["links"].append(index)
        port["link"] = index
        self.api[str(destination)]["inputs"][name] = [str(source), output_slot]

    def group(self, title, bounds):
        self.groups.append({"id": len(self.groups) + 1, "title": title, "bounding": list(bounds), "color": "#354957", "font_size": 22, "locked": False})

    def save(self, filename):
        self.validate()
        ui = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/jeankassio/ComfyUI_YuE2/" + filename)),
            "revision": 0, "version": 1,
            "state": {"lastGroupId": len(self.groups), "lastNodeId": len(self.nodes), "lastLinkId": len(self.links), "lastRerouteId": 0},
            "nodes": self.nodes, "links": self.links, "groups": self.groups, "reroutes": [], "config": {},
            "extra": {"ds": {"scale": .65, "offset": [40, 80]}, "info": {
                "name": self.name, "author": "Jean Kássio", "description": "Official YuE2 Python pipeline with pinned unquantized models and FLAC output.",
                "version": "0.1.0", "created": "2026-10-12", "modified": "2026-10-12", "software": "ComfyUI",
            }},
        }
        for extension, content in (("json", ui), ("api.json", self.api)):
            path = EXAMPLES / f"{filename}.{extension}"
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{filename}: {len(self.nodes)} nodes, {len(self.links)} links")

    def validate(self):
        for node in self.nodes:
            if node["type"] in CLASSES:
                schema = CLASSES[node["type"]].INPUT_TYPES()
                api_inputs = self.api[str(node["id"])]["inputs"]
                for name, definition in schema.get("required", {}).items():
                    if name not in api_inputs:
                        raise ValueError(f"Missing required input: {node['type']}.{name}")
                    value = api_inputs[name]
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in self.api:
                        continue
                    if isinstance(definition[0], list) and value not in definition[0]:
                        raise ValueError(f"Invalid choice: {node['type']}.{name}")
        for link in self.links:
            origin = self.nodes[link["origin_id"] - 1]["outputs"][link["origin_slot"]]
            target = self.nodes[link["target_id"] - 1]["inputs"][link["target_slot"]]
            if link["id"] not in origin["links"] or target["link"] != link["id"]:
                raise ValueError("Workflow link metadata is inconsistent")


def loader(workflow, position=(50, 60)):
    return workflow.add("SomoraYuE2ModelLoader", position, (340, 330), {"device": "cuda:0", "backend": "torch-eager", "quantization": "none", "vae": "YuE2-Vae", "offload_ar": False})


def text(workflow, song, position=(450, 60), **changes):
    values = {"style": song["style"], "lyrics": song["lyrics"], "cot": "full", "seed": song["seed"], "automatic_cfg": True, "song_id": song["id"], **changes}
    return workflow.add("SomoraYuE2TextEncode", position, (480, 630), values)


def main():
    EXAMPLES.mkdir(exist_ok=True)
    song = json.loads((EXAMPLES / "source-song.json").read_text(encoding="utf-8"))
    score = (EXAMPLES / "original-melody.abc").read_text(encoding="utf-8")

    workflow = Workflow("YuE2 Official — Text to Music")
    model, request = loader(workflow), text(workflow, song)
    generate = workflow.add("SomoraYuE2Generate", (990, 60), (310, 150))
    save = workflow.add("SaveAudioAdvanced", (1360, 60), (340, 240), {"filename_prefix": "audio/YuE2/text-to-music", "format": "flac"})
    workflow.connect(model, 0, generate, "models")
    workflow.connect(request, 0, generate, "text")
    workflow.connect(generate, 0, save, "audio")
    workflow.group("01 — Official models", (25, 15, 390, 420))
    workflow.group("02 — Style and lyrics", (425, 15, 530, 720))
    workflow.group("03 — Generate and listen", (965, 15, 760, 340))
    workflow.save("01-text-to-music")

    workflow = Workflow("YuE2 Official — Staged Generation")
    model, request = loader(workflow), text(workflow, song)
    options = workflow.add("SomoraYuE2GenerationOptions", (50, 490), (340, 510))
    plan = workflow.add("SomoraYuE2Plan", (990, 60), (320, 180))
    semantic = workflow.add("SomoraYuE2SemanticTokens", (990, 360), (320, 160))
    synthesize = workflow.add("SomoraYuE2Synthesize", (1380, 360), (320, 160))
    decode = workflow.add("SomoraYuE2Decode", (1770, 360), (320, 160))
    save = workflow.add("SaveAudioAdvanced", (2160, 360), (340, 240), {"filename_prefix": "audio/YuE2/staged", "format": "flac"})
    workflow.connect(request, 0, plan, "text")
    workflow.connect(plan, 0, semantic, "plan")
    workflow.connect(semantic, 0, synthesize, "semantic")
    workflow.connect(synthesize, 0, decode, "latents")
    workflow.connect(decode, 0, save, "audio")
    for node in (plan, semantic, synthesize, decode):
        workflow.connect(model, 0, node, "models")
        workflow.connect(options, 0, node, "options")
    workflow.group("01 — Models and sampling controls", (25, 15, 390, 1040))
    workflow.group("02 — Song request", (425, 15, 530, 720))
    workflow.group("03 — Exact plan → semantic tokens → acoustic latents → audio", (965, 15, 1560, 640))
    workflow.save("02-staged-generation")

    workflow = Workflow("YuE2 Official — Editable ABC Score")
    model = loader(workflow)
    abc = workflow.add("SomoraYuE2ABCInput", (450, 60), (480, 420), {"abc": score}, title="Edit this original ABC composition")
    request = text(workflow, song, position=(990, 60), song_id="edited_city_lights")
    options = workflow.add("SomoraYuE2GenerationOptions", (50, 490), (340, 510))
    generate = workflow.add("SomoraYuE2Generate", (1530, 60), (320, 150))
    save = workflow.add("SaveAudioAdvanced", (1910, 60), (340, 240), {"filename_prefix": "audio/YuE2/edited-score", "format": "flac"})
    workflow.connect(abc, 0, request, "abc")
    workflow.connect(model, 0, generate, "models")
    workflow.connect(request, 0, generate, "text")
    workflow.connect(options, 0, generate, "options")
    workflow.connect(generate, 0, save, "audio")
    workflow.group("01 — Models and sampling controls", (25, 15, 390, 1040))
    workflow.group("02 — Edit the score, then describe the performance", (425, 15, 1090, 720))
    workflow.group("03 — Render the edited composition", (1505, 15, 770, 340))
    workflow.save("03-edited-abc-score")

    workflow = Workflow("YuE2 Official — Audio to Cover")
    model = loader(workflow)
    source = workflow.add("LoadAudio", (450, 60), (350, 200), {"audio": "select-source-audio.wav"}, title="Upload and select your source song")
    transcribe = workflow.add("SomoraYuE2Transcribe", (450, 350), (350, 350), {"mode": "melody", "device": "cuda:0", "dtype": "bf16", "preset": "default"})
    request = text(workflow, song, position=(860, 60), cot="melody", song_id="city_lights_cover", style="English, jazz-funk, warm expressive lead vocal, Rhodes piano, rounded bass and lively drums, 88 BPM")
    options = workflow.add("SomoraYuE2GenerationOptions", (50, 490), (340, 510))
    generate = workflow.add("SomoraYuE2Generate", (1400, 60), (320, 150))
    save = workflow.add("SaveAudioAdvanced", (1780, 60), (340, 240), {"filename_prefix": "audio/YuE2/cover", "format": "flac"})
    workflow.connect(source, 0, transcribe, "audio")
    workflow.connect(transcribe, 0, request, "abc")
    workflow.connect(model, 0, generate, "models")
    workflow.connect(request, 0, generate, "text")
    workflow.connect(options, 0, generate, "options")
    workflow.connect(generate, 0, save, "audio")
    workflow.group("01 — Models and sampling controls", (25, 15, 390, 1040))
    workflow.group("02 — Source recording and melody transcription", (425, 15, 400, 740))
    workflow.group("03 — Target style and lyrics", (835, 15, 530, 720))
    workflow.group("04 — Generate a new interpretation", (1375, 15, 770, 340))
    workflow.save("04-audio-to-cover")


if __name__ == "__main__":
    main()
