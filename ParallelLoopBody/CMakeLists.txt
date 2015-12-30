cmake_minimum_required(VERSION 2.8)

PROJECT (ParallelLoopBodySample)


file(GLOB ParallelLoopBodySample_SRCS
    "*.h"
    "*.cpp"
)
ADD_EXECUTABLE (ParallelLoopBodySample ${ParallelLoopBodySample_SRCS})

if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( ParallelLoopBodySample ${OpenCV_LIBS} )
endif (OpenCV_FOUND)
