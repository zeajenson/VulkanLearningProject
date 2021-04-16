cmake -G Ninja
ninja

glslc shaders/shader.frag -o frag.spv
glslc shaders/shader.vert -o vert.spv

