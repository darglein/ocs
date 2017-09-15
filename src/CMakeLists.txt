

FILE(GLOB main_SRC  *.cpp)
FILE(GLOB saiga_SRC  saiga/*.cpp)
FILE(GLOB_RECURSE PROG_HEADER  *.h)


SET(PROG_SRC ${main_SRC})


FILE(GLOB_RECURSE CUDA_SRC  cuda/*.cu)
FILE(GLOB_RECURSE SAIGA_CUDA_SRC  saiga/*.cu)
cuda_compile(cuda_objs ${CUDA_SRC} ${SAIGA_CUDA_SRC})
SET(PROG_SRC ${PROG_SRC} ${cuda_objs})

#creates 'folders' in the solution explorer of visual studio
include(source_group_helper)
GroupSources2(${PROJECT_SOURCE_DIR}/src)


include_directories(.)




add_executable(${PROG_NAME} ${PROG_SRC} ${saiga_SRC} ${PROG_HEADER})
target_link_libraries(${PROG_NAME} ${LIBS} )



