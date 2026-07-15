extends CanvasLayer

@onready var luna: CharacterBody2D = %Luna


func _ready() -> void:
	visible = false


func show_screen() -> void:
	%SlideViewer.visible = false
	luna.visible = false
	luna.set_physics_process(false)
	visible = true


func _on_restart_button_pressed() -> void:
	get_tree().reload_current_scene()
