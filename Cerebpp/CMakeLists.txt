project(Cerebpp)

set(UTILITY
    src/time.cpp
    src/datapoint.cpp
    include/Crb/Utility/vec2.hpp
    include/Crb/Utility/time.hpp
    include/Crb/Utility/datapoint.hpp
)

set(NN
    src/nn.cpp
    src/layer.cpp
    include/Crb/NN/nn.hpp
)

set(SYSTEM
    src/file.cpp
    src/keyboard.cpp
    src/mouse.cpp
    include/Crb/System/file.hpp
    include/Crb/System/keyboard.hpp
    include/Crb/System/mouse.hpp
)

add_library(Cerebpp STATIC ${UTILITY} ${NN} ${SYSTEM})

target_include_directories(Cerebpp PUBLIC include)

source_group("Utility" FILES ${UTILITY})
source_group("NN" FILES ${NN})
source_group("System" FILES ${SYSTEM})
