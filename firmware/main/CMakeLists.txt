# Edit following two lines to set component requirements (see docs)
set(COMPONENT_REQUIRES )
set(COMPONENT_PRIV_REQUIRES )

set(COMPONENT_SRCS "main.cc"
	"models/face_detection_model_settings.cc"
	"models/face_detection_model_data.cc"
	"image_utils.c")
set(COMPONENT_ADD_INCLUDEDIRS "")
set(PRIV_REQUIRES tflite-lib)
register_component()