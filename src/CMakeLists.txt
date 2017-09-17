

FILE(GLOB main_SRC  *.cpp)

if(NOT SAIGA_FOUND)
	FILE(GLOB_RECURSE saiga_SRC  dep/saiga/*.cpp)
	SET(main_SRC ${main_SRC} ${saiga_SRC})
	include_directories(dep)
endif()

FILE(GLOB_RECURSE PROG_HEADER  *.h)


SET(PROG_SRC ${main_SRC})


FILE(GLOB CUDA_SRC  *.cu)
if(NOT SAIGA_FOUND)
	FILE(GLOB_RECURSE SAIGA_CUDA_SRC  dep/saiga/*.cu)
	SET(CUDA_SRC ${CUDA_SRC} ${SAIGA_CUDA_SRC})
endif()

cuda_compile(cuda_objs ${CUDA_SRC})
SET(PROG_SRC ${PROG_SRC} ${cuda_objs})

#creates 'folders' in the solution explorer of visual studio
include(source_group_helper)
GroupSources2(${PROJECT_SOURCE_DIR}/src)


include_directories(.)




add_executable(${PROG_NAME} ${PROG_SRC} ${PROG_HEADER})
target_link_libraries(${PROG_NAME} ${LIBS} )



