extends CharacterBody2D

const SPEED = 250.0
const JUMP_FORCE = -450.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")
var jumps_left = 2   # cuántos saltos le quedan (nuevo)

func _ready() -> void:
	set_physics_process(false)

func _physics_process(delta):
	if not is_on_floor():
		velocity.y += gravity * delta
	else:
		jumps_left = 2   # al tocar piso, recarga los saltos (nuevo)

	if Input.is_action_just_pressed("jump") and jumps_left > 0:
		velocity.y = JUMP_FORCE
		jumps_left -= 1   # gasta un salto (nuevo)

	var direction = Input.get_axis("move_left", "move_right")
	velocity.x = direction * SPEED

	move_and_slide()

	if direction != 0:
		$AnimatedSprite2D.flip_h = direction < 0
		$AnimatedSprite2D.play("Walk")
	else:
		$AnimatedSprite2D.play("Idle")
