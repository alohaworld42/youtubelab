"""Shot-Spezifikation + Prompt-Bauer.

Shots werden pro Run nach runs/<id>/shots.json geschrieben (default = die 3
Beweis-Shots + Kontrolle). Seeds sind fest: Kind-Seed (cast.json) + idx*100,
Gruppe = base_seed + Summe der seed_offsets. Kontrolle s01_norefs = Seed von s01.
"""
import json
import os

STYLE = ("A high-quality 3D CGI toon animation in a preschool show style, soft rounded characters, "
         "bright clean colors, big friendly eyes, gentle daylight.")

DEFAULT_SHOTS = [
    {"id": "s01", "cast": ["zuri"], "shot_type": "medium",
     "scene": "a sunny green park with a red bench, colorful trees and a blue sky",
     "action": "she claps her hands and sings happily, looking at the camera",
     "camera": "medium shot at her eye level, camera holds steady",
     "engines": ["K1", "K2", "V2"]},
    {"id": "s02", "cast": ["zuri"], "shot_type": "medium",
     "scene": "a bright cozy kitchen with a wooden table, a bowl of red apples and a window with sunlight",
     "action": "she waves hello with one hand and smiles, then holds up a small red toy star",
     "camera": "medium shot at her eye level, slow gentle push-in",
     "engines": ["K1", "K2", "V2"]},
    {"id": "s03", "cast": ["zuri", "kofi", "nala"], "shot_type": "wide",
     "scene": "the same sunny green park with the red bench, colorful trees and a blue sky",
     "action": "the three children stand at different distances from the camera, Kofi a step in front, "
               "Zuri to his left slightly behind, Nala to the right, all clapping and singing together",
     "camera": "wide shot, camera holds steady",
     "engines": ["K1", "K2", "V2"]},
    {"id": "s01_norefs", "cast": ["zuri"], "shot_type": "medium", "control_of": "s01",
     "scene": "a sunny green park with a red bench, colorful trees and a blue sky",
     "action": "she claps her hands and sings happily, looking at the camera",
     "camera": "medium shot at her eye level, camera holds steady",
     "engines": ["V2"], "norefs": True},
]

NEGATIVE_SHORT = "text, watermark, logo, blurry, distorted faces, extra limbs"


def load_cast(lab_dir):
    with open(os.path.join(lab_dir, "cast", "cast.json"), "r", encoding="utf-8") as f:
        cast = json.load(f)
    by_id = {c["id"]: c for c in cast["characters"]}
    return cast, by_id


def write_default_shots(path):
    if os.path.isfile(path):
        return path
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"shots": DEFAULT_SHOTS}, f, indent=2, ensure_ascii=False)
    return path


def load_shots(path, only=None):
    with open(path, "r", encoding="utf-8") as f:
        shots = json.load(f)["shots"]
    if only:
        shots = [s for s in shots if s["id"] in only]
    return shots


def shot_seed(shot, cast, by_id, shots_all):
    """Deterministic seed per shot. Kontroll-Shot erbt den Seed seines Originals."""
    if shot.get("control_of"):
        orig = next(s for s in shots_all if s["id"] == shot["control_of"])
        return shot_seed(orig, cast, by_id, shots_all)
    idx = int("".join(ch for ch in shot["id"] if ch.isdigit()) or "0")
    if len(shot["cast"]) == 1:
        return by_id[shot["cast"][0]]["seed"] + idx * 100
    return cast["base_seed"] + sum(by_id[c]["seed_offset"] for c in shot["cast"]) + idx * 100


def count_clause(names):
    """Positiv formuliert: LLM-Textencoder (H3) rendern Negationen oft mit."""
    n = len(names)
    if n == 1:
        return (f"{names[0]} is the only person in the entire scene, alone in an otherwise empty setting; "
                f"the background is free of people.")
    return (f"The scene contains exactly these {n} kids and nobody else: {', '.join(names)}; "
            f"the background is free of other people.")


def keyframe_prompt(shot, by_id):
    """Prompt fuer Klein / Qwen-Edit: 'Image N is <Name> (...)' + Szene."""
    kids = [by_id[c] for c in shot["cast"]]
    refs = " ".join(f"Image {i + 1} is {k['name']}, a {k['wardrobe']}." for i, k in enumerate(kids))
    names = [k["name"] for k in kids]
    return (f"{refs} Using exactly these {len(kids)} child{'ren' if len(kids) > 1 else ''} from the reference images "
            f"with their exact faces, skin tones, hairstyles, body proportions and outfits, place them in a new scene: "
            f"{STYLE} Setting: {shot['scene']}. Action: {shot['action']}. {shot['camera']}. "
            f"{count_clause(names)}")


def h3_prompt(shot, by_id, with_refs=True):
    """Prompt fuer MiniMaxH3ReferenceToVideo. Tokenizer setzt <Picture i> in Verbindungsreihenfolge."""
    kids = [by_id[c] for c in shot["cast"]]
    names = [k["name"] for k in kids]
    if with_refs:
        refs = " ".join(f"<Picture {i + 1}> is {k['name']}, a {k['wardrobe']}." for i, k in enumerate(kids))
        keep = ("Keep each child's face, hair, skin tone, body proportions and outfit exactly as in their picture "
                "throughout the whole video. ")
    else:
        refs = " ".join(f"{k['name']} is a {k['wardrobe']}." for k in kids)
        keep = ""
    return (f"{STYLE} {refs} {keep}{count_clause(names)} "
            f"Setting: {shot['scene']}. Action: {shot['action']}. Camera: {shot['camera']}. "
            f"Audio: soft outdoor ambience and gentle cheerful music, no speech.")
