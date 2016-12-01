
#ifndef INCLUDE_UTILS_H_
#define INCLUDE_UTILS_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <limits>
#include "cat.h"

namespace myccg {
namespace utils {

using cat::Cat;
using cat::CatPair;

const std::string DropBrackets(const std::string& in);

int FindClosingBracket(const std::string& in, int start);

int FindNonNestedChar(const std::string& haystack, const std::string& needles);

std::vector<std::string> Split(const std::string& line, char delim);

std::string
trim(const std::string& string, const char* trimCharacterList=" \t\v\r\n");

std::unordered_map<Cat, std::vector<Cat>>
LoadUnary(const std::string& filename);

std::vector<Cat>
LoadCategoryList(const std::string& filename);

template<typename T> int ArgMax(T* from, T* to) {
    T max_val = std::numeric_limits<T>::lowest();
    int max_idx = -1, i = 0;
    while (from != to) {
        if (max_val <= *from) {
            max_idx = i;
            max_val = *from;
        }
        i++; from++;
    }
    return max_idx;
}

template<typename T> int ArgMin(T* from, T* to) {
    T min_val = std::numeric_limits<T>::max();
    int min_idx = -1, i = 0;
    while (from != to) {
        if (min_val >= *from) {
            min_idx = i;
            min_val = *from;
        }
        i++; from++;
    }
    return min_idx;
}

void ReplaceAll(std::string* target,
        const std::string& from, const std::string& to);


std::unordered_set<CatPair>
LoadSeenRules(const std::string& filename);

} // namespace utils
} // namespace myccg

#endif
