# Step 4 — Ensemble station (5 slides + interactive demo)

Godot project: C:\Users\arii_\Documents\Godot\prueba

## The folder
res://Slides/Portal4/ contains 6 PNGs:
- slide_01.png .. slide_05.png → normal presentation slides (navigate with arrows)
- slide_06.png → the DEMO BACKGROUND. It is NOT a normal slide. It's the artwork behind
  the interactive classifier (it has an empty text field and button areas drawn on it).

## Behaviour
Oracle1 opens the viewer on res://Slides/Portal4/.
- Slides 1 to 5 behave normally: left/right arrows, X to close.
- When the user reaches slide_06.png (the last one), the viewer must switch to DEMO MODE:
  - Hide the right arrow (nothing after this).
  - Keep the left arrow so the user can go back to the slides.
  - HIDE the X button. The user must not close it that way.
  - Show the interactive controls ON TOP of slide_06.png:
      · a LineEdit where the user types a comment
      · a "CLASIFICAR" button
      · a Label showing the result
      · a "FINALIZAR" button
  - Position these controls to sit inside the areas drawn on slide_06.png. Tell me the
    coordinates you pick so I can adjust them.
- The user can classify as many comments as they want: type, press CLASIFICAR, read the
  result, retype, classify again. The window stays open.
- The result Label clears when the user types a new comment.
- FINALIZAR → ends the game and shows the Final Screen (built in step 5).

## Classification logic
Reuse the existing clasificar_falso() function from oracle_1.gd (simulated keyword matching).
Do NOT change that logic — I'll swap it for a real backend call later. Keep it isolated in
one function so it's easy to replace.

## Styling (match the game)
Buttons: background #C9184A, white text, 2px #FFF0F3 border, hard 4px offset shadow in
#590D22 (no blur), hover #FF4D6D.

## Important
- Unique node names, no shared resources.
- The old OracleUI CanvasLayer can be removed or repurposed — tell me what you did.

Show me the plan first. Apply without intermediate confirmation, then STOP so I can test.
