
#include "parser.h"
#include "cmdline.h"
#include "grammar.h"
#include "parser_tools.h"
#include <signal.h>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace myccg;

Parser* parser;
Tagger* tagger;
std::string model;
std::string lang;
bool use_dependency;
bool unidirectional;
float beta;
int prune_size;
Comparator comp;
std::unordered_set<Cat> root_cats;
LogLevel loglevel;

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
                              comp,
                              unidirectional ? En::headfirst_binary_rules : En::dep_binary_rules,
                              beta, prune_size, loglevel);
        } else {
            parser = new DepAStarParser<Ja>(
                          tagger, model, root_cats,
                              comp,
                              unidirectional ? Ja::headfinal_binary_rules : Ja::binary_rules,
                              beta, prune_size, loglevel);
        }
    } else {
        if (lang == "en") {
            parser = new AStarParser<En>(
                          tagger, model, root_cats,
                              comp, En::binary_rules, beta, prune_size, loglevel);
        } else {
            parser = new AStarParser<Ja>(
                          tagger, model, root_cats,
                              comp, Ja::binary_rules, beta, prune_size, loglevel);
        }
    }
}

void sig_handler(int signo) {
    if (signo == SIGTERM || signo == SIGINT)
        Parser::keep_going = false;
}

int main(int argc, char const* argv[])
{

    signal(SIGTERM, sig_handler);
    signal(SIGINT, sig_handler);

    cmdline::parser p;
    p.add<std::string>("model", 'm', "model directory");
    p.add<std::string>("format", 'f', "output format [auto,ja,xml,deriv,conll]", false, "auto",
            cmdline::oneof<std::string>("auto", "deriv", "xml", "ja", "conll"));
    p.add<float>("beta", 'b', "beta for pruning", false, 0.0000001);
    p.add<int>("pruning", 'p', "pruning size", false, 50);
    p.add<std::string>("lang", 'l', "language [en,ja]", true, "en");
    p.add<std::string>("input", 'i', "input file", false, "");
    p.add("dep", 'd', "use dependency scores");
    p.add("no-seen-rules", '\0', "suppress the use of seen rules");
    p.add("no-cat-dict", '\0', "suppress the use of category dictionary");
    p.add("no-beta", '\0', "suppress the use of beta for filtering");
    p.add("format-simple", '\0', "output trees without feature values");
    p.add("uni", '\0', "head first grammar");
    p.add("debug", '\0', "debugging");
    p.add("help", 'h', "print help");

    if ( !p.parse(argc, argv) || p.exist("help") ) {
        std::cerr << p.error_full() << p.usage();
        return 0;
    }

    model = p.get<std::string>("model");
    lang = p.get<std::string>("lang");
    root_cats = lang == "ja" ? Ja::possible_root_cats : En::possible_root_cats;
    unidirectional = p.exist("uni");
    use_dependency = p.exist("dep");
    comp = !use_dependency && lang == "ja" ? JapaneseComparator : NormalComparator; // LongerDependencyComparator;
    beta = p.get<float>("beta");
    prune_size = p.get<int>("pruning");
    loglevel = p.exist("debug") ? Debug : Info;

#ifdef _OPENMP
    std::cerr << "OpenMP : On, threads = " << omp_get_max_threads() << std::endl;
    if ( p.exist("debug") )
        omp_set_num_threads(1);
#endif

    LoadTagger();
    tagger->SetEnv(argv[0]);

    LoadParser();

    if ( ! p.exist("no-seen-rules") )
        parser->LoadSeenRules();

    if ( ! p.exist("no-cat-dict") )
        parser->LoadCategoryDict();

    if ( p.exist("no-beta") )
        parser->SetUseBeta(false);

    // Load inputs from stdin
    std::string input;
    std::vector<std::string> inputs;
    if ( p.exist("input") && ! p.get<std::string>("input").empty() ) {
        std::ifstream ifs( p.get<std::string>("input") );
        if(ifs.fail()) {
            std::cerr << "File do not exist: "
                      << p.get<std::string>("input")
                      << std::endl;
            exit(0);
        }
        while (std::getline(ifs, input))
            inputs.push_back(input);

    } else {
        while (std::getline(std::cin, input))
            inputs.push_back(input);
    }

    // Parse
    auto res = parser->Parse(inputs);

    // Output
    if (p.get<std::string>("format") == "xml") {
        ToXML(res, !p.exist("format-simple"));
    } else if (p.get<std::string>("format") == "deriv") {
        for (auto&& tree: res)
            std::cout << Derivation(tree, !p.exist("format-simple")) << std::endl;
    } else if (p.get<std::string>("format") == "conll") {
        for (auto&& tree: res)
            std::cout << CoNLL(tree) << std::endl;

    } else if (p.get<std::string>("format") == "ja") {
        for (auto&& tree: res)
            std::cout << JaCCG(tree) << std::endl;

    } else if (p.get<std::string>("format") == "auto") {
        for (unsigned i = 0; i < res.size(); i++)
            std::cout << "ID=" << i+1 << std::endl << res[i].get() << std::endl;
    }
    
    return 0;
}
