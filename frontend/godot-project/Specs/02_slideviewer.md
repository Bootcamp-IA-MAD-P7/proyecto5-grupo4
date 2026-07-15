# Step 2 — Build a reusable slide viewer

Godot project: C:\Users\arii_\Documents\Godot\prueba
Main scene: Main.tscn

## Goal
ONE reusable slide viewer used by all three portals. Do NOT build separate UI per portal.
Existing per-portal SlideUI CanvasLayers (SlideUI, SlideUI2, SlideUI3...) should be replaced
by this single viewer.

## What to build
A CanvasLayer scene named SlideViewer with:
- A TextureRect filling the screen showing the current slide image
- A LEFT ARROW button (previous slide)
- A RIGHT ARROW button (next slide)
- An X button (close) at top right
- A small page indicator (e.g. "3 / 9")

## Script API (slide_viewer.gd)
- `open(folder_path: String)` → loads ALL PNGs from that folder in alphabetical order,
  shows slide 1, makes the viewer visible, hides Luna and calls
  set_physics_process(false) on the gameplay Luna.
- Right arrow → next slide. Disabled/hidden on the last slide.
- Left arrow → previous slide. Disabled/hidden on the first slide.
- X button → close viewer, show Luna again, set_physics_process(true) so the player
  keeps playing.
- Starts hidden (visible = false).

## Styling (match the game palette)
- Arrow buttons and X: background #C9184A, text/icon white, border 2px #FFF0F3,
  hard offset shadow 4px in #590D22, no blur.
- Hover state #FF4D6D.
- Page indicator text in #FFCCD5.
- Place the arrows at the bottom left and bottom right of the slide area, the X at top right.

## Important
- The slide PNGs are 960x540. The game viewport may differ, so the TextureRect must scale
  the image to fit without distortion (keep aspect).
- Give every node a unique name. No shared resources between nodes.
- The gameplay Luna (CharacterBody2D) is the one to hide/pause. Don't touch other Luna sprites.

Show me the plan before starting. Apply all changes of this step without asking for
intermediate confirmation, then STOP so I can test in Godot.
If you edit .tscn as text, remind me to reload from disk.
