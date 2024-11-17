project(Cerebpp)

set(UTILITY
    src/time.cpp
    src/datapoint.cpp
    include/Crb/Utility/vec2.hpp
    include/Crb/Utility/time.hpp
    include/Crb/Utility/datapoint.hpp
)

set(FNN
    src/fnn.cpp
    src/layer.cpp
    include/Crb/FNN/fnn.hpp
    include/Crb/FNN/layer.hpp
)

set(SYSTEM
    src/file.cpp
    src/keyboard.cpp
    src/mouse.cpp
    include/Crb/System/file.hpp
    include/Crb/System/keyboard.hpp
    include/Crb/System/mouse.hpp
)

add_library(Cerebpp STATIC ${UTILITY} ${FNN} ${SYSTEM})

target_include_directories(Cerebpp PUBLIC include)

source_group("Utility" FILES ${UTILITY})
source_group("FNN" FILES ${FNN})
source_group("System" FILES ${SYSTEM})
