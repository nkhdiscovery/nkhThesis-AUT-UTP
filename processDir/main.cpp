#include <QCoreApplication>
#include <cstdio>
#include <iostream>
#include "boost/filesystem.hpp"

#include "errorcodes.h"

/****************** nkhStart: macro utils ******************/
#define errStr(x) #x

#define checkPath(p) if(!exists(p)){\
    cerr << p << " : " << errStr(NO_SUCH_FILE_OR_DIR\n);\
    return NO_SUCH_FILE_OR_DIR;}


/****************** nkhEnd: macro utils ******************/


using namespace std;
using namespace boost::filesystem;

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    if (argc == 3)
    {
        path inputDir(argv[1]), outDir(argv[2]);
        checkPath(inputDir);
        checkPath(outDir);

        return NORMAL_STATE;
    }
    else
    {
        cerr << errStr(error: INSUFFICIENT_ARGUMENTS\n);
        return INSUFFICIENT_ARGUMENTS;
    }
    //return a.exec();
}

