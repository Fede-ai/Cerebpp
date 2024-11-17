project(Test)

set(SOURCE
	main.cpp
)

add_executable(Test ${SOURCE})

target_link_libraries(Test PRIVATE Cerebpp)

target_include_directories(Test PRIVATE ${CMAKE_SOURCE_DIR}/Cerebpp/include)