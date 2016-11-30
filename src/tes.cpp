#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>

#define print(x) std::cout << (x) << std::endl;

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
std::vector<std::pair<std::string, std::string>> Parse(const std::string& input) {
    std::vector<std::pair<std::string, std::string>> res;
    std::istringstream s(input);
    std::string pair;
    while (std::getline(s, pair, ',')) {
        int eq = pair.find("=");
        std::string key = pair.substr(0, eq);
        std::string value = pair.substr(eq + 1);
        res.emplace_back(key, value);
    }
    return res;
}

void test(const std::string& input) {
    auto res = Parse(input);
    for (auto& pair: res) {
        std::cout << pair.first << " = " << pair.second << std::endl;
    }
        std::cout << std::endl;
}

int main(int argc, char const* argv[])
{
    std::string ex1 = "mod=nm,form=base,fin=f";
    std::string ex2 = "case=ga,mod=nm,fin=f";
    std::string ex3 = "case=to,mod=nm,fin=f";
    std::string ex4 = "case=ni,mod=nm,fin=f";
    test(ex1);
    test(ex2);
    test(ex3);
    test(ex4);

    
    return 0;
}
