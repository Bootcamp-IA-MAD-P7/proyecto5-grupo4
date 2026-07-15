extends Area2D

func _ready():
	$Green.play("Green")

func _on_body_entered(body: Node2D) -> void:
	if body.name == "Luna":
		%SlideViewer.open("res://Slides/Portal1")
