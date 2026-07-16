# Step 8 — Connect the demo to the real backend

Godot project: C:\Users\arii_\Documents\Godot\prueba

## Goal
Replace the simulated `clasificar_falso()` function with a real HTTP call to the deployed
FastAPI model. This affects BOTH ensemble stations (OracleIntro at the start and Oracle1
at the end) — they both have the interactive demo.

## The API

URL: https://sexism-detector-1000036994845.europe-west1.run.app/predict
Method: POST
Headers: Content-Type: application/json
Request body: {"text": "the comment the user typed"}
Response body: {"label": "sexist" | "not sexist", "confidence": 0.87}

## What to build

1. Add an `HTTPRequest` node to each demo (or one shared, whatever is cleanest —
   tell me what you chose).

2. When the user presses CLASIFICAR:
   - Read the text from the LineEdit.
   - If it's empty, show "Escribi un comentario primero" and do nothing else.
   - Show "Consultando al oraculo..." in the result Label while waiting.
   - Disable the CLASIFICAR button so the user can't spam requests.
   - Send the POST request with the body above.

3. On `request_completed`:
   - Re-enable the CLASIFICAR button.
   - Parse the JSON response.
   - Display in the result Label, formatted like this:
     "Probabilidad de sexismo: 87.0%" on one line
     "SEXISTA" or "NO SEXISTA" on the next line
     (label "sexist" → "SEXISTA", anything else → "NO SEXISTA")
     (confidence comes as 0-1, multiply by 100 for the percentage)

4. Error handling — IMPORTANT, this is a live demo:
   - If the request fails (no internet, server down, timeout, non-200 code), show a clear
     message in the Label like "No se pudo conectar con el modelo. Intenta de nuevo."
     Never let it crash or hang silently.
   - If the JSON can't be parsed, same thing.
   - Add a reasonable timeout so it doesn't hang forever.

5. Keep the old `clasificar_falso()` function in the file but unused (commented out or
   renamed), as a fallback in case the backend is down during the presentation.
   Tell me how to switch back to it quickly if I need to.

## Reference code (adapt it, don't paste blindly)
```gdscript
var url = "https://sexism-detector-1000036994845.europe-west1.run.app/predict"
var body = JSON.stringify({"text": texto})
var headers = ["Content-Type: application/json"]
http.request(url, headers, HTTPClient.METHOD_POST, body)

# on response:
var probabilidad = json.confidence * 100
var etiqueta = "SEXISTA" if json.label == "sexist" else "NO SEXISTA"
resultado_label.text = "Probabilidad de sexismo: %.1f%%\n%s" % [probabilidad, etiqueta]
```

## Important
- Apply this to BOTH OracleIntro and Oracle1. They have separate scripts — update both,
  don't merge them into one shared script.
- Put the URL in a single constant at the top of each script so it's easy to change.
- Unique node names.

Show me the plan first. Apply without intermediate confirmation, then STOP so I can test.
