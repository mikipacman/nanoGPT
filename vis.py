# Visualize midi in html
import glob
import os
import random
import string
import sys

from tqdm import tqdm

sys.path.append("../")
from scripts.txt_to_midi import single_txt_to_midi

import collections
import tempfile
from typing import Dict
import tiktoken
import torch
from midi_player import MIDIPlayer
from midi_player.stylers import dark


def html_midi_vis(midis: Dict[str, Dict[str, str]]) -> str:
    htmls = collections.defaultdict(dict)
    for row_name, row_items in midis.items():
        for item_name, item_midi_path in row_items.items():
            mp = MIDIPlayer(item_midi_path, height=2, width="5%", styler=dark, title=item_name)
            htmls[row_name][item_name] = mp.html

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script>
document.addEventListener('DOMContentLoaded', function() {{
  const collapsibles = document.querySelectorAll('.collapsible');
  
  collapsibles.forEach(function(coll) {{
    const content = coll.nextElementSibling;
    
    coll.addEventListener('click', function() {{
      content.style.display = content.style.display === 'none' ? 'block' : 'none';
    }});
  }});
}});
</script>
<style>
    .row {{
        display: flex;
    }}
    .item {{
        display: inline-block;
        margin: 5px;
        padding: 10px;
        border: 1px s
        olid #000;
        width: 400px;
    }}
    .collapsible {{
        cursor: pointer;
        user-select: none;
        padding-left: 10px;
        border: 1px solid #ccc;
    }}
    .content {{
        display: none;
        padding: 10px;
    }}
</style>
</head>

<body>
{''.join([f"<div class='collapsible'><h3>{row_name}</h3></div><div class='content'>" + ''.join([
        "<div class='item'>" + htmls[row_name][item_name] + "</div>" for item_name in htmls[row_name]])
          + "</div>" for row_name in htmls])}
</body>
</html>
"""

    return html


def file_content_to_midi(file_content, tmp):
    random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(16))
    txt_file = os.path.join(tmp, f"{random_string}.txt")
    file_content = "\n".join(file_content.split("\n")[:-1] + ["end\n"])
    open(txt_file, "w").write(file_content)
    midi_file = os.path.join(tmp, f"{random_string}.mid")
    single_txt_to_midi(txt_file, midi_file)
    return midi_file


# sample some songs for given prompts and generate html vis, Return (html, percent_success)
@torch.no_grad()
def sample_songs(model, device, prompt_dir, num_samples=3, max_new_tokens=1024, temperature=0.7, topk=200):
    # tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    with tempfile.TemporaryDirectory() as tmp:
        model.eval()
        midis = collections.defaultdict(dict)
        success = 0

        prompts = glob.glob(os.path.join(prompt_dir, "*"))
        for prompt_path in tqdm(prompts, desc="Sampling songs"):
            song_name = os.path.basename(prompt_path).rsplit(".", 1)[0]
            midis[song_name]["prompt"] = file_content_to_midi(open(prompt_path, "r").read(), tmp)
            start_id = encode(open(prompt_path, "r").read())
            x = (torch.tensor(start_id, dtype=torch.long, device=device)[None, ...])
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=topk)
            for i in range(num_samples):
                try:
                    file_content = decode(y[i].tolist())
                    midi_file = file_content_to_midi(file_content, tmp)
                except:
                    print(f"Failed to convert {song_name} to MIDI")
                    continue
                success += 1
                midis[song_name][f"sample_{i}"] = midi_file

        model.train()
        return html_midi_vis(midis), success / (num_samples * len(prompts))


if __name__ == "__main__":
    midis = {
        "cannibal corpse": {
            "prompt": "/home/mp/Projects/gpgpt/data/midi/cannibalcorpse__1ib17y9y__zerothehero.midi",
            "sample_1": "/home/mp/Projects/gpgpt/data/midi/cannibalcorpse__1ib17y9y__zerothehero.midi",
            "sample_2": "/home/mp/Projects/gpgpt/data/midi/cannibalcorpse__1ib17y9y__zerothehero.midi",
        },
        "metallica": {
            "prompt": "/home/mp/Projects/gpgpt/data/midi/metallica__...and_justice_for_all.midi",
            "sample_1": "/home/mp/Projects/gpgpt/data/midi/metallica__...and_justice_for_all.midi",
        },
    }
    html = html_midi_vis(midis)

    open("midi_vis.html", "w").write(html)
