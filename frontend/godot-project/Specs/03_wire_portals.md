# Step 3 — Wire the FOUR portals to the slide viewer

Godot project: C:\Users\arii_\Documents\Godot\prueba

## Mapping (4 portals, not 3)

- PortalGreen  → res://Slides/Portal1/  (9 slides, presentation)
- PortalYellow → res://Slides/Portal2/  (1 slide, TF-IDF model)
- PortalWhite  → res://Slides/Portal3/  (1 slide, BERT model)
- Oracle1      → res://Slides/Portal4/  (6 PNGs, the Ensemble station — see step 4,
                                          the 6th PNG is the interactive demo background)

## What to do
- In each portal's script, on body_entered with Luna, call SlideViewer.open("res://Slides/PortalN/")
  with its own folder, instead of the old per-portal SlideUI.
- Remove/disable the old SlideUI, SlideUI2, SlideUI3 CanvasLayers and their unused
  close-button logic. Clean dead code.
- Keep each portal's fire/monster animation exactly as it is. Don't touch it.

## Single-slide portals
PortalYellow and PortalWhite have only ONE slide each. In that case the viewer must hide
both arrows (there's nowhere to navigate) and show only the X to close.

## Important
- Each portal keeps its OWN .gd file. Never share one script between portals.
- Verify body_entered is connected to the right function in each portal.

Show me the plan first. Apply without intermediate confirmation, then STOP so I can test.
Remind me to reload the scene from disk if you edit .tscn as text.
