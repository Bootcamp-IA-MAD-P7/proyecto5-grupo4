extends CanvasLayer

@onready var luna: CharacterBody2D = %Luna


func _on_start_button_pressed() -> void:
	hide()
	luna.set_physics_process(true)
