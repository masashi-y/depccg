
#include <stdexcept>
#include <sstream>
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
    // for (unsigned i = 0; i < haystack.size(); i++) {
    for (unsigned i = haystack.size()-1; 0 < i; i--) {
        // std::cout << haystack <<  i << std::endl;
        // std::cout << i << std::endl;
        if (haystack[i] == ')')
            open_brackets++;
        else if (haystack[i] == '(')
            open_brackets--;
        if (open_brackets == 0)
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


void ReplaceAll(std::string* target, const std::string& from, const std::string& to) {
    std::string::size_type pos = 0;
    while(pos = target->find(from, pos), pos != std::string::npos) {
        target->replace(pos, from.length(), to);
        pos += to.length();
    }
}

unsigned int utf8_strlen(std::string str) {

    unsigned int len = 0;
    unsigned char lead; 
    unsigned char_size = 0;

    for (unsigned pos = 0; pos < str.size(); pos += char_size) {

        lead = str[pos];

        if (lead < 0x80) {
            char_size = 1;
        } else if (lead < 0xE0) {
            char_size = 2;
        } else if (lead < 0xF0) {
            char_size = 3;
        } else {
            char_size = 4;
        }

        len += 1;
    }

    return len;
}
} // namespace utils
} // namespace myccg

