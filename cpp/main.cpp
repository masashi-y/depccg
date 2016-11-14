
#include "parser.h"
#include "cmdline.h"
#include "test.h"
#include <omp.h>

void test()
{
    // myccg::tagger::test();
    // myccg::tree::test();
    myccg::parser::test();
}

using namespace myccg;

int main(int argc, char const* argv[])
{
#ifdef _OPENMP
    std::cout << "OpenMP : On, threads = " << omp_get_max_threads() << std::endl;
#endif

#ifdef TEST
    test();
#else
    cmdline::parser p;
    p.add<std::string>("model", 'm', "model directory");
    p.add("deriv", 'd', "output result in derivation format");
    p.add<float>("beta", 'b', "beta for pruning", false, 0.0000001);
    p.add("help", 'h', "print help");

    if (!p.parse(argc, argv) || p.exist("help")) {
        std::cout << p.error_full() << p.usage();
        return 0;
    }
    tagger::ChainerTagger tagger(p.get<std::string>("model"));
    parser::AStarParser parser(&tagger, p.get<std::string>("model"));
    std::string input;
    while (std::getline(std::cin, input)) {
        auto res = parser.Parse(input, p.get<float>("beta"));
        if (p.exist("deriv")) {
            tree::ShowDerivation(res);
            std::cout << std::endl;
        } else {
            std::cout << res->ToStr() << std::endl;
        }
    }
#endif
    
    return 0;
}
