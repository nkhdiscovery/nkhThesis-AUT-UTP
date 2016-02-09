QT += core
QT -= gui

TARGET = processDir
CONFIG += console
CONFIG -= app_bundle

LIBS += -lboost_system -lboost_filesystem
LIBS += -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu/ -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_cudawarping -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_imgproc -lopencv_cudaarithm -lopencv_core -lopencv_cudev
#-lopencv_contrib
QMAKE_CXXFLAGS += -std=c++11 -Wall -O3 -march=corei7-avx
QMAKE_LFLAGS += -fopenmp -pthread
QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp

INCLUDEPATH +=  -I/usr/local/cuda-7.5/targets/x86_64-linux/include/ -I/usr/local/include/opencv -I/usr/local/include 3rd/ 3rd/external

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
    3rd/external/io_png/io_png.c \
    3rd/segment/egbis/disjoint-set.cpp \
    3rd/segment/egbis/filter.cpp \
    3rd/segment/egbis/misc.cpp \
    3rd/segment/egbis/segment-graph.cpp \
    3rd/segment/egbis/segment-image.cpp \
    3rd/segment/egbis.cpp

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
    3rd/FunctionProfiling.h \
    3rd/segment/egbis/convolve.h \
    3rd/segment/egbis/disjoint-set.h \
    3rd/segment/egbis/filter.h \
    3rd/segment/egbis/image.h \
    3rd/segment/egbis/imconv.h \
    3rd/segment/egbis/imutil.h \
    3rd/segment/egbis/misc.h \
    3rd/segment/egbis/pnmfile.h \
    3rd/segment/egbis/segment-graph.h \
    3rd/segment/egbis/segment-image.h \
    3rd/segment/egbis.h \
    nkhAlgorithms.h \
    egbiscv.h
