
#include <chrono>
#include "parser.h"
#include "cmdline.h"
#include "grammar.h"
#include "parser_tools.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace myccg;

int main(int argc, char const* argv[])
{
#ifdef _OPENMP
    std::cerr << "OpenMP : On, threads = " << omp_get_max_threads() << std::endl;
#endif

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
    const std::string& model = p.get<std::string>("model");

#ifdef JAPANESE
    ChainerDependencyTagger tagger(p.get<std::string>("model"));
    DepAStarParser<Ja> parser(&tagger,
          utils::LoadUnary(model + "/unary_rules.txt"),
          utils::LoadSeenRules(model + "/seen_rules.txt"),
          Ja::possible_root_cats,
          JapaneseComparator);
#else
    ChainerTagger tagger(p.get<std::string>("model"));
    AStarParser<En> parser(&tagger,
          utils::LoadUnary(model + "/unary_rules.txt"),
          utils::LoadSeenRules(model + "/seen_rules.txt"),
          En::possible_root_cats,
          LongerDependencyComparator);
#endif
    std::string input;
    std::vector<std::string> inputs;
    while (std::getline(std::cin, input))
        inputs.push_back(input);
    start = std::chrono::system_clock::now();
    auto res = parser.Parse(inputs);
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
    if (p.get<std::string>("format") == "xml") {
        ToXML(res);
    } else if (p.get<std::string>("format") == "deriv") {
        for (auto&& tree: res) {
            std::cout << Derivation(tree) << std::endl;
        }
    } else if (p.get<std::string>("format") == "auto") {
        for (auto&& tree: res) {
            std::cout << tree.get() << std::endl;
        }
    }
    std::cerr << "elapsed time: " << elapsed << " seconds" << std::endl;
    
    return 0;
}
