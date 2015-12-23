#include <QCoreApplication>
#include <cstdio>
#include <iostream>
#include "errorcodes.h"
#include "macros.h"

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    if (argc == 3)
    {
        printf("Goood\n");
        return NORMAL_STATE;
    }
    else
    {
        std::cerr << errStr(error: INSUFFICIENT_ARGUMENTS\n);
        return INSUFFICIENT_ARGUMENTS;
    }
    //return a.exec();
}

