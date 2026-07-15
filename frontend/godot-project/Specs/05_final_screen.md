# Step 5 — Final screen

Godot project: C:\Users\arii_\Documents\Godot\prueba

## Goal
A Final Screen that appears when the user presses FINALIZAR in the Ensemble demo.
It also appears if the player keeps walking right past the Ensemble to the end of the map.

## What to build
- A CanvasLayer named FinalScreen, starts hidden.
- A TextureRect filling the screen with the background image from res://FinalScreen/
  (I'll drop the PNG there — find whatever PNG is in that folder).
- An AnimatedSprite2D with Luna, using the existing "Hi Luna" frames already in the project
  (same ones the START screen uses). Looping and autoplay so she waves.
  Position her in the left area, roughly x 55-255, y 175-410 of a 960x540 layout.
- Same structure and approach as the START screen — mirror it.

## Trigger logic
1. FINALIZAR button in the Ensemble demo → show FinalScreen, hide everything else,
   stop gameplay (set_physics_process(false) on the gameplay Luna).
2. Also: if Luna walks past the Ensemble to the far right edge of the level, trigger the
   same FinalScreen automatically. Add an Area2D at the end of the map for this.

## Important
- The FinalScreen Luna is decorative (AnimatedSprite2D only, no physics). It is NOT the
  gameplay Luna. Don't confuse them.
- Unique node names, no shared resources.

Show me the plan before starting. Apply without intermediate confirmation, then STOP.
