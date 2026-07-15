# Step 6 — Duplicate the Ensemble station at the START of the level

Godot project: C:\Users\arii_\Documents\Godot\prueba

## Why
The tutor wants the interactive demo to be the FIRST thing the player sees. So we add a
second Ensemble station near the beginning of the level, before PortalGreen. It is a
duplicate of the existing Oracle1 station — same monster, same visuals, same logic,
including the FINALIZAR button. Only the slide folder changes.

## What to build

### The new station
- New Area2D named `OracleIntro`, placed on the GROUND near the start of the level,
  BEFORE PortalGreen, so Luna walks into it right away. Mandatory, on the walking path.
- Same visuals as Oracle1: the ensemble monster AnimatedSprite2D (reuse the same frames
  already in the project) plus its fire portal behind it, exactly like Oracle1 looks now.
- Its own CollisionShape2D.

### Its script — CRITICAL
- `OracleIntro` must have its OWN script file: `oracle_intro.gd`. Do NOT attach or share
  `oracle_1.gd`. They point to different folders, sharing the script will break both.
  (This exact bug has bitten this project before.)
- Copy the full logic from `oracle_1.gd` (open the SlideViewer, demo mode on the last PNG,
  the classifier controls, FINALIZAR), changing only the folder it opens.

### Its folder
- Opens `res://Slides/EnsembleIntro/` which contains 3 PNGs:
  - slide_01.png → El problema
  - slide_02.png → Nuestro objetivo
  - slide_03.png → the DEMO BACKGROUND (same artwork as Portal4's demo background)
- Behaviour is identical to Oracle1's station: slides 1 and 2 navigate with arrows,
  slide_03 switches to DEMO MODE with LineEdit, CLASIFICAR, result Label and FINALIZAR,
  and the X is hidden in demo mode.

### Portal1 changes
`res://Slides/Portal1/` loses its first two slides (they moved to EnsembleIntro).
I will re-export and renumber those PNGs myself — you don't need to touch the files.
Just make sure the SlideViewer loads whatever PNGs are in the folder alphabetically,
without hardcoding a count.

## Verify before finishing
Confirm the final mapping is:
- OracleIntro   → res://Slides/EnsembleIntro/  (start of level, mandatory)
- PortalGreen   → res://Slides/Portal1/
- PortalYellow  → res://Slides/Portal2/  (TF-IDF)
- PortalWhite   → res://Slides/Portal3/  (BERT)
- Oracle1       → res://Slides/Portal4/  (Ensemble, end)

## Important
- Unique node names. OracleIntro and Oracle1 must NOT share scripts, SpriteFrames
  resources, or collision shapes. Make every resource unique.
- Both stations have FINALIZAR and both go to the FinalScreen. That's intended.

Show me the plan first. Apply without intermediate confirmation, then STOP so I can test.
Remind me to reload the scene from disk if you edit .tscn as text.
