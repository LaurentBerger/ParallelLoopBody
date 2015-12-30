cmake_minimum_required(VERSION 2.8)


PROJECT (ParallelLoopBodyOpenCV)
find_package(OpenCV  REQUIRED)

FOREACH( mesdossiers 
        ParallelLoopBody
        SpeedTest
    )

    ADD_SUBDIRECTORY(${mesdossiers})

ENDFOREACH()


