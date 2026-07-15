extends CanvasLayer

signal classify_requested(text: String)
signal finalize_requested()

@onready var background: TextureRect = $Background
@onready var left_arrow: Button = $LeftArrow
@onready var right_arrow: Button = $RightArrow
@onready var close_button: Button = $CloseButton
@onready var page_indicator: Label = $PageIndicator
@onready var demo_line_edit: LineEdit = $DemoLineEdit
@onready var demo_classify_button: Button = $DemoClassifyButton
@onready var demo_classify_button_shadow: ColorRect = $DemoClassifyButtonShadow
@onready var demo_result_label: Label = $DemoResultLabel
@onready var demo_finalize_button: Button = $DemoFinalizeButton
@onready var demo_finalize_button_shadow: ColorRect = $DemoFinalizeButtonShadow
@onready var demo_continue_button: Button = $DemoContinueButton
@onready var demo_continue_button_shadow: ColorRect = $DemoContinueButtonShadow
@onready var luna: CharacterBody2D = %Luna

var slides: Array[Texture2D] = []
var index: int = 0
var demo_mode_enabled: bool = false
var demo_continue_enabled: bool = false


func _ready() -> void:
	visible = false
	_set_demo_controls_visible(false)


func open(folder_path: String, demo_mode_on_last_slide: bool = false, show_continue_in_demo: bool = false) -> void:
	slides.clear()
	demo_mode_enabled = demo_mode_on_last_slide
	demo_continue_enabled = show_continue_in_demo

	var dir := DirAccess.open(folder_path)
	if dir == null:
		push_error("SlideViewer: could not open folder %s" % folder_path)
		return

	var file_names: Array[String] = []
	dir.list_dir_begin()
	var file_name := dir.get_next()
	while file_name != "":
		if not dir.current_is_dir() and file_name.get_extension().to_lower() == "png":
			file_names.append(file_name)
		file_name = dir.get_next()
	dir.list_dir_end()
	file_names.sort()

	for name in file_names:
		var tex := load(folder_path.path_join(name)) as Texture2D
		if tex:
			slides.append(tex)

	index = 0
	_update_slide()

	luna.visible = false
	luna.set_physics_process(false)
	visible = true


func _is_demo_slide() -> bool:
	return demo_mode_enabled and index == slides.size() - 1


func _update_slide() -> void:
	if slides.is_empty():
		return
	background.texture = slides[index]

	var in_demo := _is_demo_slide()

	left_arrow.visible = index > 0
	right_arrow.visible = not in_demo and index < slides.size() - 1
	close_button.visible = not in_demo
	page_indicator.visible = not in_demo
	page_indicator.text = "%d / %d" % [index + 1, slides.size()]

	_set_demo_controls_visible(in_demo)
	if in_demo:
		demo_line_edit.text = ""
		demo_result_label.text = ""


func _set_demo_controls_visible(v: bool) -> void:
	demo_line_edit.visible = v
	demo_classify_button.visible = v
	demo_classify_button_shadow.visible = v
	demo_result_label.visible = v
	demo_finalize_button.visible = v
	demo_finalize_button_shadow.visible = v
	demo_continue_button.visible = v and demo_continue_enabled
	demo_continue_button_shadow.visible = v and demo_continue_enabled


func set_classification_result(text: String) -> void:
	demo_result_label.text = text


func _on_left_arrow_pressed() -> void:
	if index > 0:
		index -= 1
		_update_slide()


func _on_right_arrow_pressed() -> void:
	if index < slides.size() - 1:
		index += 1
		_update_slide()


func _on_close_button_pressed() -> void:
	visible = false
	luna.visible = true
	luna.set_physics_process(true)


func _on_demo_classify_button_pressed() -> void:
	classify_requested.emit(demo_line_edit.text)


func _on_demo_line_edit_text_changed(_new_text: String) -> void:
	demo_result_label.text = ""


func _on_demo_finalize_button_pressed() -> void:
	finalize_requested.emit()
