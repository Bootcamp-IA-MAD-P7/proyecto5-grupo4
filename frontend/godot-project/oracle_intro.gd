extends Area2D

func _ready():
	$Monster.play("default")
	%SlideViewer.classify_requested.connect(_on_slide_viewer_classify_requested)
	%SlideViewer.finalize_requested.connect(_on_slide_viewer_finalize_requested)

func _on_body_entered(body: Node2D) -> void:
	if body.name == "Luna":
		%SlideViewer.open("res://Slides/EnsembleIntro", true, true)

func _on_slide_viewer_classify_requested(texto: String) -> void:
	%SlideViewer.set_classification_result(clasificar_falso(texto))

func clasificar_falso(texto: String) -> String:
	var palabras = ["mujer", "cocina", "callate", "tonta", "inferior"]
	for p in palabras:
		if p in texto.to_lower():
			return "SEXISTA (score simulado: 0.87)"
	return "NO SEXISTA (score simulado: 0.12)"

func _on_slide_viewer_finalize_requested() -> void:
	%FinalScreen.show_screen()
