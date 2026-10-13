# Moves the push-constant block of a SPIRV-Cross MSL kernel to [[buffer(8)]].
# --msl-decoration-binding maps storage buffers to their Vulkan binding
# numbers (0-4) and leaves the push-constant block on buffer(0) as well.
if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "INPUT and OUTPUT are required")
endif()
file(READ "${INPUT}" source)
string(REGEX REPLACE "(constant [A-Za-z_0-9]+& [A-Za-z_0-9]+ )\\[\\[buffer\\(0\\)\\]\\]" "\\1[[buffer(8)]]" patched "${source}")
if(patched STREQUAL source)
  message(FATAL_ERROR "${INPUT}: push-constant block on buffer(0) not found")
endif()
file(WRITE "${OUTPUT}.tmp" "${patched}")
file(RENAME "${OUTPUT}.tmp" "${OUTPUT}")
