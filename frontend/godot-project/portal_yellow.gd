extends Area2D

func _ready():
	$Yellow.play("Yellow")

func _on_body_entered(body: Node2D) -> void:
	if body.name == "Luna":
		%SlideViewer.open("res://Slides/Portal2")
