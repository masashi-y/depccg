
#include <stdexcept>

#ifndef INCLUDE_DEBUG_H_
#define INCLUDE_DEBUG_H_

#define print(value) std::cout << (value) << std::endl;
#define NO_IMPLEMENTATION { throw std::runtime_error(__PRETTY_FUNCTION__); }
#define GREEN(str) std::cerr << "\033[32m" << str << "\033[93m" << std::endl
#define RED(str) std::cerr << "\033[31m" << str << "\033[93m" << std::endl
#define BLUE(str) std::cerr << "\033[34m" << str << "\033[93m" << std::endl
#define CYAN(str) std::cerr << "\033[36m" << str << "\033[93m" << std::endl
#define POPPED RED("********POPPED********")
#define ACCEPT GREEN("********ACCEPT********")
#define TREETYPE(str) BLUE("********" << str << "********")
#define DEBUG(x) std::cerr << #x": " << (x) << std::endl

#endif

