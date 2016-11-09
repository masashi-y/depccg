
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

typedef std::pair<const myccg::cat::Category*, const myccg::cat::Category*> CatPair;

const std::string drop_brackets(const std::string& in);

int find_closing_bracket(const std::string& in, int start);

int find_non_nested_char(const std::string haystack, const std::string needles);

std::vector<std::string> split(const std::string& line, char delim);

std::string
trim(const std::string& string, const char* trimCharacterList=" \t\v\r\n");

std::unordered_map<const cat::Category*, std::vector<const cat::Category*>>
load_unary(const std::string& filename);

std::vector<const cat::Category*>
load_category_list(const std::string& filename);

template<typename T> int ArgMax(T* from, T* to) {
    T max_val = std::numeric_limits<T>::lowest();
    int max_idx, i = 0;
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
    int min_idx, i = 0;
    while (from != to) {
        if (min_val >= *from) {
            min_idx = i;
            min_val = *from;
        }
        i++; from++;
    }
    return min_idx;
}

std::string ReplaceAll(const std::string target,
        const std::string from, const std::string to);

struct hash_cat_pair
{
    inline size_t operator () (const CatPair& p) const {
        return ((p.first->GetId() << 31) | (p.second->GetId()));
    }
};

std::unordered_set<CatPair, hash_cat_pair>
load_seen_rules(const std::string& filename);

void test();

} // namespace utils
} // namespace myccg

#endif
