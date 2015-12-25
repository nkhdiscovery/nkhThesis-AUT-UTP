#ifndef NKHUTIL_H
#define NKHUTIL_H

#include <cstdio>
#include <iostream>
#include <ctime>
#include <limits.h>
#include <cstring>
#include <chrono>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;

#include <vector>
#include <algorithm>
using namespace std;


enum{
NORMAL_STATE,
INSUFFICIENT_ARGUMENTS,
NO_SUCH_FILE_OR_DIR,
NOT_A_DIR,
NOT_REGULAR_FILE,

};


/****************** nkhStart: macro utils ******************/
#define getStr(x) #x

#define checkPath(p) if(!exists(p)){\
    cerr << p << " : " << getStr(NO_SUCH_FILE_OR_DIR\n);\
    return NO_SUCH_FILE_OR_DIR; }
#define checkDir(p) checkPath(p);\
    if(!is_directory(p)){\
    cerr << p << " : " << getStr(NOT_A_DIR\n);\
    return NOT_A_DIR; }
#define checkRegularFile(p) checkPath(p);\
    if(!is_regular_file(p)){\
    cerr << p << " : " << getStr(NOT_REGULAR_FILE\n);\
    return NOT_REGULAR_FILE; }
/****************** nkhEnd: macro utils ******************/

/************************* nkhStart: timer template *************************
 * Usage:
 * cout << measure<std::chrono::milliseconds>(functionName, args...) << endl;
 * also can be used for void functions like
 * measure<std::chrono::nanoseconds>(someVoidFunc, args...);
 ****************************************************************************/
template<typename T> string time_type()                  { return "unknown";      }
template<> string time_type<std::chrono::nanoseconds >() { return "nanoseconds";  }
template<> string time_type<std::chrono::microseconds>() { return "microseconds"; }
template<> string time_type<std::chrono::milliseconds>() { return "milliseconds"; }
template<> string time_type<std::chrono::seconds     >() { return "seconds";      }
template<> string time_type<std::chrono::minutes     >() { return "minutes";      }
template<> string time_type<std::chrono::hours       >() { return "hours";        }
template <typename TimeT, typename Functor, typename ... Args>
auto measure(Functor f, Args && ... args)
    -> decltype(f(std::forward<Args>(args)...))
{
    struct scoped_timer
    {
        scoped_timer() : now_(std::chrono::high_resolution_clock::now()) {}
        ~scoped_timer()
        {
            auto elapsed = std::chrono::duration_cast<TimeT
                >(std::chrono::high_resolution_clock::now() - now_).count();
            std::cout << "Time elapsed: " << elapsed << " in " <<  time_type<TimeT>() << std::endl;
        }
        private:
            std::chrono::high_resolution_clock::time_point const now_;
    } scoped_timer;

    return f(std::forward<Args>(args)...);
}
/************************* nkhEnd: timer template *************************/

#endif // NKHUTIL_H
