# Step 7 — Two final buttons

Godot project: C:\Users\arii_\Documents\Godot\prueba

## A) "CONTINUAR" button in OracleIntro's demo mode

The intro ensemble station (OracleIntro) currently forces the user to either press
FINALIZAR (which ends the game) or navigate back to a previous slide and press the X.
That's bad — the intro demo is a hook, the user should be able to leave it and keep playing.

Add a "CONTINUAR" button in OracleIntro's DEMO MODE, next to FINALIZAR.
- CONTINUAR → closes the whole viewer, shows the gameplay Luna again, calls
  set_physics_process(true) on her, and lets the player keep walking toward PortalGreen.
  Exactly what the X does on normal slides.
- FINALIZAR stays as it is (goes to the FinalScreen).
- This applies ONLY to OracleIntro. Oracle1 (the ensemble at the end) keeps only
  FINALIZAR, no CONTINUAR.

## B) "VOLVER A EMPEZAR" button on the FinalScreen

Add a button labeled "VOLVER A EMPEZAR" on the FinalScreen.
- Pressing it returns to the START screen, so the player can play again from the top.
- Reset the game state properly: the gameplay Luna goes back to her starting position,
  the FinalScreen hides, the StartScreen shows, and the game waits on START again.
  Reloading the current scene is fine if that's the cleanest way (get_tree().reload_current_scene()),
  as long as it lands on the START screen and not mid-game.
- Place it centered at the bottom of the FinalScreen, below the team panel.

## Styling for both buttons (match the game)
Background #C9184A, white text, 2px #FFF0F3 border, hard 4px offset shadow in #590D22
(no blur), hover #FF4D6D.

## Important
- Unique node names. Don't touch the other portals or the classification logic.

Show me the plan first. Apply without intermediate confirmation, then STOP so I can test.
Remind me to reload the scene from disk if you edit .tscn as text.
