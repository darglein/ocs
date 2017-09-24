cmake_minimum_required(VERSION 2.8)

set(PROG_NAME ocs)
project (${PROG_NAME})

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE relwithdebinfo)


set (PROGRAM_VERSION_MAJOR 1)
set (PROGRAM_VERSION_MINOR 0)

option (CUDA_DEBUG "enables CUDA_SYNC_CHECK_ERROR and CUDA_ASSERT" OFF) 

############# Required LIBRARIES ###############

############# CUDA ###############

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
	SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
	SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})
	message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}")

	#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
	set(CUDA_PROPAGATE_HOST_FLAGS ON)
	set(CUDA_SEPARABLE_COMPILATION OFF)

	list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=compute_30 )
	list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52 )

	list( APPEND CUDA_NVCC_FLAGS -use_fast_math)

	#disables an eigen warning
	list( APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

	if(BUILD_SHARED)
	      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -DSAIGA_DLL_EXPORTS)
	endif()

	if(UNIX)
	      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC)
	endif()
endif()




message(STATUS "NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")


#saiga
find_package(SAIGA)
if(SAIGA_FOUND)
	include_directories(${SAIGA_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${SAIGA_LIBRARY})
	set(OCS_USE_SAIGA 1)
	add_definitions(-DHAS_SAIGA)
endif()


#opencv
find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
SET(LIBS ${LIBS} ${OpenCV_LIBRARIES})
message(STATUS "OpenCV: ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBRARIES}")
set(OCS_USE_OPENCV 1)

############# COMPILER FLAGS ###############


if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-long-long")
endif(UNIX)

if(MSVC)
	#multiprocessor compilation for visual studio
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") 
endif()

if(CUDA_DEBUG)
	add_definitions(-DCUDA_DEBUG)
else()
	add_definitions(-DCUDA_NDEBUG)
endif()

############# Generate config file ###############

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/src/buildconfig.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/buildconfig.h"
)



############# SOURCE ###############

add_subdirectory(src)

#############  output dir ############# 

if(MSVC)
	#visual studio working dir
	set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

#copies executable to top level source tree
set_target_properties( ${PROG_NAME}
	PROPERTIES
    	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")


