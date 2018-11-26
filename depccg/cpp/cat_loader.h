
#ifndef INCLUDE_CAT_LOADER_H_
#define INCLUDE_CAT_LOADER_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "cat.h"

namespace myccg {
namespace utils {


std::unordered_map<Cat, std::vector<Cat>>
LoadUnary(const std::string& filename);

std::vector<Cat>
LoadCategoryList(const std::string& filename);

std::unordered_map<std::string, std::vector<bool>>
LoadCategoryDict(const std::string& filename, const std::vector<Cat>& targets);

std::unordered_set<CatPair>
LoadSeenRules(const std::string& filename, Cat (*Preprocess)(Cat));

} // namespace utils
} // namespace myccg

#endif
