extends Area2D

const API_URL := "https://sexism-detector-1000036994845.europe-west1.run.app/predict"
const USE_FALLBACK_CLASSIFIER := false

@onready var classifier_request: HTTPRequest = $ClassifierRequest

func _ready():
	$Monster.play("default")
	classifier_request.timeout = 60.0
	classifier_request.request_completed.connect(_on_classifier_request_completed)
	%SlideViewer.classify_requested.connect(_on_slide_viewer_classify_requested)
	%SlideViewer.finalize_requested.connect(_on_slide_viewer_finalize_requested)

func _on_body_entered(body: Node2D) -> void:
	if body.name == "Luna":
		%SlideViewer.open("res://Slides/EnsembleIntro", true, true, self)

func _on_slide_viewer_classify_requested(texto: String) -> void:
	if %SlideViewer.current_owner != self:
		return

	if texto.strip_edges() == "":
		%SlideViewer.set_classification_result("Escribi un comentario primero")
		return

	if USE_FALLBACK_CLASSIFIER:
		%SlideViewer.set_classification_result(clasificar_falso_fallback(texto))
		return

	%SlideViewer.set_classify_button_disabled(true)
	%SlideViewer.set_classification_result("Despertando al oraculo... (puede tardar unos segundos)")

	var body := JSON.stringify({"text": texto})
	var headers := ["Content-Type: application/json"]
	var err := classifier_request.request(API_URL, headers, HTTPClient.METHOD_POST, body)
	if err != OK:
		print("OracleIntro classifier: request() failed to start, error code %d" % err)
		%SlideViewer.set_classify_button_disabled(false)
		%SlideViewer.set_classification_result("No se pudo conectar con el modelo. Intenta de nuevo.")

func _on_classifier_request_completed(result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if %SlideViewer.current_owner != self:
		return

	%SlideViewer.set_classify_button_disabled(false)

	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		print("OracleIntro classifier: request failed - result=%d response_code=%d body=%s" % [result, response_code, body.get_string_from_utf8()])
		%SlideViewer.set_classification_result("No se pudo conectar con el modelo. Intenta de nuevo.")
		return

	var json := JSON.new()
	if json.parse(body.get_string_from_utf8()) != OK:
		print("OracleIntro classifier: JSON parse failed - response_code=%d body=%s" % [response_code, body.get_string_from_utf8()])
		%SlideViewer.set_classification_result("No se pudo conectar con el modelo. Intenta de nuevo.")
		return

	var data = json.data
	if typeof(data) != TYPE_DICTIONARY or not data.has("label") or not data.has("confidence"):
		print("OracleIntro classifier: response missing expected fields - response_code=%d body=%s" % [response_code, body.get_string_from_utf8()])
		%SlideViewer.set_classification_result("No se pudo conectar con el modelo. Intenta de nuevo.")
		return

	var probabilidad: float = float(data["confidence"]) * 100.0
	var etiqueta := "SEXISTA" if data["label"] == "sexist" else "NO SEXISTA"
	%SlideViewer.set_classification_result("Probabilidad de sexismo: %.1f%%\n%s" % [probabilidad, etiqueta])

func clasificar_falso_fallback(texto: String) -> String:
	var palabras = ["mujer", "cocina", "callate", "tonta", "inferior"]
	for p in palabras:
		if p in texto.to_lower():
			return "SEXISTA (score simulado: 0.87)"
	return "NO SEXISTA (score simulado: 0.12)"

func _on_slide_viewer_finalize_requested() -> void:
	if %SlideViewer.current_owner != self:
		return
	%FinalScreen.show_screen()
