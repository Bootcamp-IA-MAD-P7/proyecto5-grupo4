extends Area2D

func _ready():
	$White.play("default")

func _on_body_entered(body: Node2D) -> void:
	if body.name == "Luna":
		%SlideViewer.open("res://Slides/Portal3")
