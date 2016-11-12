
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <fstream>
#include <regex>
#include "utils.h"

namespace myccg {
namespace utils {


const std::string drop_brackets(const std::string& in) {
    int length = in.size();
    if (in[0] == '(' && in[length-1] == ')' &&
            find_closing_bracket(in, 0) == length - 1)
        return in.substr(1, length - 2);
    else
        return in;

}

int find_closing_bracket(const std::string& in, int start) {
    int open_brackets = 0;
    for (int i = 0; i < in.size(); i++) {
        if (in[i] == '(')
            open_brackets++;
        else if (in[i] == ')')
            open_brackets--;
        if (open_brackets == 0)
            return i;
    }
    throw std::runtime_error("Mismatched brackets in string");
}

int find_non_nested_char(const std::string haystack, const std::string needles) {
    int open_brackets = 0;
    for (int i = 0; i < haystack.size(); i++) {
        if (haystack[i] == '(')
            open_brackets++;
        else if (haystack[i] == ')')
            open_brackets--;
        else if (open_brackets == 0)
            for (int j = 0; j < needles.size(); j++) {
                if (needles[j] == haystack[i])
                    return i;
            }
    }
    return -1;
}

std::vector<std::string> split(const std::string& line, char delim) {
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


CatMap<std::vector<Cat>>
load_unary(const std::string& filename) {
    auto res = CatMap<std::vector<Cat>>();
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("failed to open: " + filename);
    std::string line, buf;
    const cat::Category *from, *to;
    std::vector<std::string> items;
    int comment;

    while (getline(in, line)) {
        comment = line.find("#");
        if (comment > -1)
            line = line.substr(0, comment);
        
        line = trim(line);
        if (line.size() == 0) continue;
        // items = split(line, ' ');
        std::stringstream ss(line);
        ss >> buf;
        from = cat::parse(buf);
        ss >> buf;
        to = cat::parse(buf);
        if (res.count(from) == 0)
            res[from] = std::vector<Cat>();
        res[from].push_back(to);
    }
    return res;
}

std::vector<Cat>
load_category_list(const std::string& filename) {
    auto res = std::vector<Cat>();
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("failed to open: " + filename);
    std::string line, buf;
    const cat::Category *ca;
    int comment;

    while (getline(in, line)) {
        comment = line.find("#");
        if (comment > -1)
            line = line.substr(0, comment);
        
        line = trim(line);
        if (line.size() == 0) continue;
        std::stringstream ss(line);
        ss >> buf;
        ca = cat::parse(buf);
        res.push_back(ca);
    }
    return res;
}

std::string ReplaceAll(const std::string target,
        const std::string from, const std::string to) {
    std::string result = target;
    std::string::size_type pos = 0;
    while(pos = result.find(from, pos), pos != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

// const std::regex feat("\\[nb\\]|\\[X\\]");
const std::string empty("");

std::unordered_set<CatPair, hash_cat_pair>
load_seen_rules(const std::string& filename) {
    auto res = std::unordered_set<CatPair, hash_cat_pair>();
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("failed to open: " + filename);
    std::string line, buf;
    const cat::Category *ca1, *ca2;
    int comment;

    while (getline(in, line)) {
        comment = line.find("#");
        if (comment > -1)
            line = line.substr(0, comment);
       
        line = trim(line);
        if (line.size() == 0) continue;
        std::stringstream ss(line);
        ss >> buf;
        // buf = std::regex_replace(buf, feat, empty);
        ca1 = cat::parse(buf);
        ss >> buf;
        // buf = std::regex_replace(buf, feat, empty);
        ca2 = cat::parse(buf);
        auto p = CatPair(ca1, ca2);
        res.insert(p);
    }
    return res;
}

void test() {
    std::cout << "----" << __FILE__ << "----" << std::endl;

    auto set = utils::load_seen_rules("seenRules");
    print(set.size());
    int i = 0;
    for (auto& elem: set) {
        std::cout << elem.first->ToStr() << ", " << elem.second->ToStr() << std::endl;
        if (++i > 10)
            break;

    }
}

} // namespace utils
} // namespace myccg

