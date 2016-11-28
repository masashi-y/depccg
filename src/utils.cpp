
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <fstream>
#include <regex>
#include "utils.h"

namespace myccg {
namespace utils {


const std::string DropBrackets(const std::string& in) {
    int length = in.size();
    if (in[0] == '(' && in[length-1] == ')' &&
            FindClosingBracket(in, 0) == length - 1)
        return in.substr(1, length - 2);
    else
        return in;

}

int FindClosingBracket(const std::string& in, int start) {
    int open_brackets = 0;
    for (unsigned i = 0; i < in.size(); i++) {
        if (in[i] == '(')
            open_brackets++;
        else if (in[i] == ')')
            open_brackets--;
        if (open_brackets == 0)
            return i;
    }
    throw std::runtime_error("Mismatched brackets in string");
}

int FindNonNestedChar(const std::string& haystack, const std::string& needles) {
    int open_brackets = 0;
    for (unsigned i = 0; i < haystack.size(); i++) {
        if (haystack[i] == '(')
            open_brackets++;
        else if (haystack[i] == ')')
            open_brackets--;
        else if (open_brackets == 0)
            for (unsigned j = 0; j < needles.size(); j++) {
                if (needles[j] == haystack[i])
                    return i;
            }
    }
    return -1;
}

std::vector<std::string> Split(const std::string& line, char delim) {
    std::istringstream iss(line);
    std::string tmp;
    std::vector<std::string> res;
    while (getline(iss, tmp, delim)) res.push_back(tmp);
    return res;
}


std::string
trim(const std::string& string, const char* trimCharacterList) {
    std::string::size_type left = string.find_first_not_of(trimCharacterList);
    if (left != std::string::npos) {
        std::string::size_type right = string.find_last_not_of(trimCharacterList);
        return string.substr(left, right - left + 1);
    }
    return string;
}


std::unordered_map<Cat, std::vector<Cat>>
LoadUnary(const std::string& filename) {
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
        Cat from = cat::Parse(buf);
        ss >> buf;
        Cat to = cat::Parse(buf);
        if (res.count(from) == 0)
            res[from] = std::vector<Cat>();
        res[from].push_back(to);
    }
    return res;
}

std::vector<Cat>
LoadCategoryList(const std::string& filename) {
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
        Cat ca = cat::Parse(buf);
        res.push_back(ca);
    }
    return res;
}

std::string ReplaceAll(const std::string& target,
        const std::string& from, const std::string& to) {
    std::string result = target;
    std::string::size_type pos = 0;
    while(pos = result.find(from, pos), pos != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

std::unordered_set<CatPair>
LoadSeenRules(const std::string& filename) {
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
        Cat ca1 = cat::Parse(buf)->StripFeat();
        ss >> buf;
        Cat ca2 = cat::Parse(buf)->StripFeat();
        auto p = CatPair(ca1, ca2);
        res.insert(p);
    }
    return res;
}

} // namespace utils
} // namespace myccg

