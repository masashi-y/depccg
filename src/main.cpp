
#include <chrono>
#include "parser.h"
#include "cmdline.h"
#include "grammar.h"
#include "parser_tools.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace myccg;

Parser* parser;
Tagger* tagger;
std::string model;
std::string lang;
bool use_dependency;
float beta;
int prune_size;
Comparator comp;
std::unordered_set<Cat> root_cats;

void LoadTagger() {
    if ( use_dependency ) {
        std::cerr << "loading dependency tagger" << std::endl;
        tagger = new ChainerDependencyTagger(model);
    } else
        tagger = new ChainerTagger(model);
}

void LoadParser() {
    if ( use_dependency ) {
        std::cerr << "loading dependency parser" << std::endl;
        if (lang == "en") {
            parser = new DepAStarParser<En>(
                          tagger, model, root_cats,
                              comp, beta, prune_size);
        } else {
            parser = new DepAStarParser<Ja>(
                          tagger, model, root_cats,
                              comp, beta, prune_size);
        }
    } else {
        if (lang == "en") {
            parser = new AStarParser<En>(
                          tagger, model, root_cats,
                              comp, beta, prune_size);
        } else {
            parser = new AStarParser<Ja>(
                          tagger, model, root_cats,
                              comp, beta, prune_size);
        }
    }
}

int main(int argc, char const* argv[])
{
#ifdef _OPENMP
    std::cerr << "OpenMP : On, threads = " << omp_get_max_threads() << std::endl;
#endif

    cmdline::parser p;
    std::chrono::system_clock::time_point start, end;
    p.add<std::string>("model", 'm', "model directory");
    p.add<std::string>("format", 'f', "output format", false, "auto",
            cmdline::oneof<std::string>("auto", "deriv", "xml"));
    p.add<float>("beta", 'b', "beta for pruning", false, 0.0000001);
    p.add<int>("pruning", 'p', "pruning size", false, 50);
    p.add<std::string>("lang", 'l', "language [en, ja]", true, "en");
    p.add("dep", 'd', "use dependency scores");
    p.add("seen-rules", '\0', "use seen rules");
    p.add("category-dict", '\0', "use category dictionary");
    p.add("help", 'h', "print help");

    if ( !p.parse(argc, argv) || p.exist("help") ) {
        std::cout << p.error_full() << p.usage();
        return 0;
    }

    model = p.get<std::string>("model");
    lang = p.get<std::string>("lang");
    comp = lang == "ja" ? JapaneseComparator : LongerDependencyComparator;
    root_cats = lang == "ja" ? Ja::possible_root_cats : En::possible_root_cats;
    use_dependency = p.exist("dep");
    beta = p.get<float>("beta");
    prune_size = p.get<int>("pruning");

    LoadTagger();
    LoadParser();

    if ( p.exist("seen-rules") )
        parser->LoadSeenRules();

    if ( p.exist("category-dict") )
        parser->LoadCategoryDict();

    std::string input;
    std::vector<std::string> inputs;
    while (std::getline(std::cin, input))
        inputs.push_back(input);
    start = std::chrono::system_clock::now();
    auto res = parser->Parse(inputs);
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
