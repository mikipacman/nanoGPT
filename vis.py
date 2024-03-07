# Visualize midi in html
import collections
from typing import Dict

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
