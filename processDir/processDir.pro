QT += core
QT -= gui

TARGET = processDir
CONFIG += console
CONFIG -= app_bundle

LIBS += -lboost_system -lboost_filesystem

DEFINES += "WITH_CUDA=1"

TEMPLATE = app

SOURCES += main.cpp

HEADERS += \
    errorcodes.h

