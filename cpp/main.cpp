
#include "parser.h"
#include "cmdline.h"

void test()
{
    myccg::tagger::test();
    myccg::tree::test();
    myccg::utils::test();
    myccg::combinator::test();
    myccg::parser::test();
}

using namespace myccg;

int main(int argc, char const* argv[])
{
#ifdef TEST
    test();
#endif
    cmdline::parser p;
    p.add<std::string>("model", 'm', "model directory");
    p.add("deriv", 'd', "output result in derivation format");
    p.add("help", 'h', "print help");

    if (!p.parse(argc, argv) || p.exist("help")) {
        std::cout << p.error_full() << p.usage();
        return 0;
    }
    tagger::ChainerTagger tagger(p.get<std::string>("model"));
    parser::AStarParser parser(&tagger, p.get<std::string>("model"));
    std::string input;
    while (std::getline(std::cin, input)) {
        auto res = parser.Parse(input);
        if (p.exist("deriv"))
            tree::ShowDerivation(res);
        else
            std::cout << res->ToStr() << std::endl;
    }
    
    return 0;
}
