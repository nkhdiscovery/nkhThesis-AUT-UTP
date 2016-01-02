QT += core
QT -= gui

TARGET = processDir
CONFIG += console
CONFIG -= app_bundle

LIBS += -lboost_system -lboost_filesystem
LIBS += -L/usr/local/lib -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_cudawarping -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_imgproc -lopencv_cudaarithm -lopencv_core -lopencv_cudev -lopencv_contrib


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
