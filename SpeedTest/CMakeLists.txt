cmake_minimum_required(VERSION 2.8)

PROJECT (TestSpeed)


file(GLOB TestSpeed_SRCS
    "*.h"
    "*.cpp"
)
ADD_EXECUTABLE (TestSpeed ${TestSpeed_SRCS})

if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
#	include_directories("${OPENCV_CONFIG_PATH}/include")
	target_link_libraries( TestSpeed ${OpenCV_LIBS} )
endif (OpenCV_FOUND)
