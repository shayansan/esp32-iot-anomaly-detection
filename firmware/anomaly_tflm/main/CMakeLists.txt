idf_component_register(
    SRCS
        "main.cpp"
        "model_data.cc"
    INCLUDE_DIRS
        "."
    PRIV_REQUIRES
        esp_timer
        esp-tflite-micro
)

target_compile_features(
    ${COMPONENT_LIB}
    PUBLIC
    cxx_std_17
)