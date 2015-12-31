QT += core
QT -= gui

TARGET = processDir
CONFIG += console
CONFIG -= app_bundle

LIBS += -lboost_system -lboost_filesystem
LIBS += -L/usr/local/lib -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_shape
LIBS += -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab
LIBS += -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_objdetect
LIBS += -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc
LIBS += -lopencv_cudafilters -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm
LIBS += -lopencv_core -lopencv_hal -lopencv_cudev

QMAKE_CXXFLAGS += -std=c++11 -Wall -O3 -march=corei7-avx
QMAKE_LFLAGS += -fopenmp -pthread

INCLUDEPATH +=  -I/usr/local/include/opencv -I/usr/local/include 3rd/ 3rd/external

DEFINES += "WITH_CUDA=1"
DEFINES += "WITH_VISUALIZATION=1"

#Start FOR DTF fast code
DEFINES += DO_FUNCTION_PROFILING
QMAKE_CXXFLAGS_RELEASE += -fno-tree-vectorize
QMAKE_CXXFLAGS_DEBUG += -fno-tree-vectorize
QMAKE_CXXFLAGS += -fno-tree-vectorize
LIBS+= -lpng

#End FOR DTF fast code

TEMPLATE = app

SOURCES += main.cpp \
    guidedfilter.cpp \
    3rd/external/io_png/io_png.c

HEADERS += \
    nkhUtil.h \
    FrameObjects.h \
    PanelObject.h \
    guidedfilter.h \
    3rd/external/tclap/CmdLine.h \
    3rd/external/io_png/io_png.h \
    3rd/Image.h \
    3rd/NC.h \
    3rd/Exception.h \
    3rd/Mat2.h \
    3rd/RF.h \
    3rd/common.h\
    3rd/rdtsc.h \
    3rd/FunctionProfiling.h
