

FILE(GLOB main_SRC  *.cpp)
FILE(GLOB_RECURSE PROG_HEADER  *.h)


SET(PROG_SRC ${main_SRC})


FILE(GLOB_RECURSE CUDA_SRC  cuda/*.cu)
cuda_compile(cuda_objs ${CUDA_SRC})
SET(PROG_SRC ${PROG_SRC} ${cuda_objs})



include_directories(.)




add_executable(${PROG_NAME} ${PROG_SRC} ${PROG_HEADER})
target_link_libraries(${PROG_NAME} ${LIBS} )



