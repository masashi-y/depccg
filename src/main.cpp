
#include "parser.h"
#include "cat.h"
#include "cmdline.h"
#include "grammar.h"
#include "test.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void test()
{
    // myccg::tagger::test();
    myccg::tree::test();
    // myccg::parser::test();
}

using namespace myccg;

int main(int argc, char const* argv[])
{
#ifdef _OPENMP
    std::cerr << "OpenMP : On, threads = " << omp_get_max_threads() << std::endl;
#endif

#ifdef TEST2
    test();
#else
    cmdline::parser p;
    std::chrono::system_clock::time_point start, end;
    p.add<std::string>("model", 'm', "model directory");
    p.add("deriv", 'd', "output result in derivation format");
    p.add<std::string>("format", 'f', "output format", false, "auto",
            cmdline::oneof<std::string>("auto", "deriv", "xml"));
    p.add<float>("beta", 'b', "beta for pruning", false, 0.0000001);
    p.add("help", 'h', "print help");

    if (!p.parse(argc, argv) || p.exist("help")) {
        std::cout << p.error_full() << p.usage();
        return 0;
    }
    tagger::ChainerTagger tagger(p.get<std::string>("model"));
    const std::string& model = p.get<std::string>("model");

#ifdef JAPANESE
    parser::AStarParser parser(&tagger,
          utils::LoadUnary(model + "/unary_rules.txt"),
          grammar::ja::binary_rules,
          utils::LoadSeenRules(model + "/seen_rules.txt"),
          grammar::ja::possible_root_cats);
#else
    parser::AStarParser parser(&tagger,
          utils::LoadUnary(model + "/unary_rules.txt"),
          grammar::en::binary_rules,
          utils::LoadSeenRules(model + "/seen_rules.txt"),
          grammar::en::possible_root_cats);
#endif
    std::string input;
    std::vector<std::string> inputs;
    while (std::getline(std::cin, input))
        inputs.push_back(input);
    start = std::chrono::system_clock::now();
    auto res = parser.Parse(inputs, p.get<float>("beta"));
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
    if (p.get<std::string>("format") == "xml") {
        tree::ToXML(res);
    } else if (p.get<std::string>("format") == "deriv") {
        for (auto&& tree: res) {
            tree::ShowDerivation(tree);
            std::cout << std::endl;
        }
    } else if (p.get<std::string>("format") == "auto") {
        for (auto&& tree: res) {
            std::cout << tree->ToStr() << std::endl;
        }
    }
    std::cerr << "elapsed time: " << elapsed << " seconds" << std::endl;
#endif
    
    return 0;
}
