
#include <iostream>
#include <sstream>
#include <fstream>
#include "cat_loader.h"
#include "utils.h"

namespace myccg {
namespace utils {

Cat GetCategory(const std::string& str) {
    return Category::Parse(str)->StripFeat("[X]", "[nb]");
    // Cat cat = Category::Parse(str);
    // if (cat->IsTypeRaised() && cat->IsFunctionInto(Category::Parse("S"))) {
    //     std::cerr << cat << std::endl;
    //     std::string s(cat->ToStr());
    //     utils::ReplaceAll(&s, "S/", "S[X]/");
    //     utils::ReplaceAll(&s, "S\\", "S[X]\\");
    //     std::cerr << s << std::endl;
    //     return Category::Parse(s);
    // }
    // return cat;
}

std::unordered_set<CatPair> LoadSeenRules(const std::string& filename,
                                          Cat (*Preprocess)(Cat)) {
    auto res = std::unordered_set<CatPair>();
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("failed to open: " + filename);
    std::string line, buf;

    while (getline(in, line)) {
        int comment = line.find("#");
        if (comment > -1)
            line = line.substr(0, comment);
       
        line = trim(line);
        if (line.size() == 0) continue;
        std::stringstream ss(line);
        ss >> buf;
        Cat ca1 = Preprocess(Category::Parse(buf));
        ss >> buf;
        Cat ca2 = Preprocess(Category::Parse(buf));
        auto p = CatPair(ca1, ca2);
        res.insert(p);
    }
    return res;
}

std::unordered_map<std::string, std::vector<bool>>
LoadCategoryDict(const std::string& filename, const std::vector<Cat>& targets) {
    auto idx_map = std::unordered_map<std::string, int>();
    for (unsigned i = 0; i < targets.size(); i++)
        idx_map.emplace(targets[i]->ToStr(), i);
    auto res = std::unordered_map<std::string, std::vector<bool>>();
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("failed to open: " + filename);
    std::string line, buf, word;

    while (getline(in, line)) {
        int comment = line.find("#");
        if (comment > -1)
            line = line.substr(0, comment);
        std::stringstream ss(line);
        ss >> word;
        auto tmp = std::vector<bool>(targets.size(), false);
        while (ss >> buf) {
            tmp[idx_map[buf]] = true;
        }
        res.emplace(word, tmp);
    }
    return res;
}

std::unordered_map<Cat, std::vector<Cat>> LoadUnary(const std::string& filename) {
    auto res = std::unordered_map<Cat, std::vector<Cat>>();
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("failed to open: " + filename);
    std::string line, buf;

    while (getline(in, line)) {
        int comment = line.find("#");
        if (comment > -1)
            line = line.substr(0, comment);
        
        line = trim(line);
        if (line.size() == 0) continue;
        std::stringstream ss(line);
        ss >> buf;
        Cat from = Category::Parse(buf);
        ss >> buf;
        Cat to = Category::Parse(buf);
        if (res.count(from) == 0)
            res[from] = std::vector<Cat>();
        res[from].push_back(to);
    }
    return res;
}

std::vector<Cat> LoadCategoryList(const std::string& filename) {
    auto res = std::vector<Cat>();
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("failed to open: " + filename);
    std::string line, buf;

    while (getline(in, line)) {
        int comment = line.find("#");
        if (comment > -1)
            line = line.substr(0, comment);
        
        line = trim(line);
        if (line.size() == 0) continue;
        std::stringstream ss(line);
        ss >> buf;
        Cat ca = Category::Parse(buf);
        res.push_back(ca);
    }
    return res;
}

} // namespace utils
} // namespace myccg
