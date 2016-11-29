
#include "parser.h"
#include "grammar.h"
#include <chrono>

namespace myccg {
namespace parser {

void test() {
    std::cout << "----" << __FILE__ << "----" << std::endl;

    const std::string model = "../model_425";
    tagger::ChainerTagger tagger(model);
    parser::AStarParser parser(&tagger,
          utils::LoadUnary(model + "/unary_rules.txt"),
          grammar::en::binary_rules,
          utils::LoadSeenRules(model + "/seen_rules.txt"),
          {cat::Parse("S[dcl]"), cat::Parse("S[wq]"),
            cat::Parse("S[q]"), cat::Parse("S[qem]"), cat::Parse("NP")});
          
    const std::string sent1 = "this is a new sentence .";
    const std::string sent2 = "Ed saw briefly Tom and Taro .";
    const std::string sent3 = "Darth Vador , also known as Anakin Skywalker is a fictional character .";
    // auto res = parser.Parse(sent1);
    // tree::ShowDerivation(res);
    // res = parser.Parse(sent2, 0.00001);
    // tree::ShowDerivation(res);
    // res = parser.Parse(sent3);
    // tree::ShowDerivation(res);
    // res = parser.Parse("But Mrs. Hills , speaking at a breakfast meeting of the American Chamber of Commerce in Japan on Saturday , stressed that the objective is not to get definitive action by spring or summer , it is rather to have a blueprint for action .");
    // tree::ShowDerivation(static_cast<const tree::Tree*>(res));

    std::chrono::system_clock::time_point start, end;
    std::vector<std::string> doc{sent1, sent2, sent3};
    std::vector<std::string> inputs;
    std::string in;
    while (getline(std::cin, in)) {
        inputs.push_back(in);
    }
    // sort(inputs.begin(), inputs.end(),
    //         [](const std::string& s1, const std::string& s2) {
    //         return s1.size() > s2.size(); });
    start = std::chrono::system_clock::now();
    auto res_doc = parser.Parse(inputs, 0.0001);
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
    for (auto&& tree: res_doc) {
        // std::cout << tree->ToStr() << std::endl;
        // tree::ShowDerivation(tree);
    }
    std::cout << "elapsed time: " << elapsed << " seconds" << std::endl;

}
} // namespace parser

namespace tree {
#define APPLY_BINARY(comb, left, right) new myccg::tree::Tree( \
        (comb)->Apply((left)->GetCategory(), (right)->GetCategory()), \
        (comb)->HeadIsLeft((left)->GetCategory(), (right)->GetCategory()), \
        (left), (right), (comb));

#define APPLY_UNARY(cat, child) new myccg::tree::Tree((cat), (child))

#define APPLICABLE(comb, left, right) std::cout << #comb": " << \
    (left)->GetCategory()->ToStr() << ", " << (right)->GetCategory()->ToStr() \
    << " --> " << \
    ((comb)->CanApply((left)->GetCategory(), (right)->GetCategory()) ? "OK" : "NO" )\
    << std::endl;

#define TEST(cond)    std::cout << #cond" --> " << ( (cond) ? "yes":"no") << std::endl;

void test()
{
    std::cout << "----" << __FILE__ << "----" << std::endl;

    auto fwd  = new combinator::ForwardApplication();
    auto bwd  = new combinator::BackwardApplication();
    auto Bx   = new combinator::GeneralizedBackwardComposition<0>(cat::Slashes::Fwd(), cat::Slashes::Bwd(), cat::Slashes::Fwd());
    auto conj = new combinator::Conjunction();
    auto rp   = new combinator::RemovePunctuation(false);

    const Node* leaves[] = {
        new Leaf("this",     cat::Parse("NP"),              0),
        new Leaf("is",       cat::Parse("(S[dcl]\\NP)/NP"), 1),
        new Leaf("a",        cat::Parse("NP[nb]/N"),        2),
        new Leaf("new",      cat::Parse("N/N"),             3),
        new Leaf("sentence", cat::Parse("N"),               4),
        new Leaf(".",        cat::Parse("."),               5),
    };

    const Tree* tree1 = APPLY_BINARY(fwd, leaves[3], leaves[4]);
    const Tree* tree2 = APPLY_BINARY(fwd, leaves[2], tree1);
    const Tree* tree3 = APPLY_BINARY(fwd, leaves[1], tree2);
    const Tree* tree4 = APPLY_BINARY(bwd, leaves[0], tree3);
    const Tree* tree5 = APPLY_BINARY(rp, tree4, leaves[5]);

    APPLICABLE(fwd, leaves[3], leaves[4]);
    APPLICABLE(fwd, leaves[2], tree1);
    APPLICABLE(fwd, leaves[1], tree2);
    APPLICABLE(bwd, leaves[0], tree3);
    APPLICABLE(rp, tree4, leaves[5]);

    print(tree5->ToStr());
    ShowDerivation(tree5);

    const Node* leaves2[] = {
        new Leaf("Ed",      cat::Parse("N"),                0),
        new Leaf("saw",     cat::Parse("(S[dcl]\\NP)/NP"),  1),
        new Leaf("briefly", cat::Parse("(S\\NP)\\(S\\NP)"), 2),
        new Leaf("Tom",     cat::Parse("N"),                3),
        new Leaf("and",     cat::Parse("conj"),             4),
        new Leaf("Taro",    cat::Parse("N"),                5),
        new Leaf(".",       cat::Parse("."),                6),
    };
    const Tree* tree2_1 = APPLY_UNARY(cat::Parse("NP"), leaves2[0]); // Ed NP
    const Tree* tree2_2 = APPLY_UNARY(cat::Parse("NP"), leaves2[3]); // Tom NP
    const Tree* tree2_3 = APPLY_UNARY(cat::Parse("NP"), leaves2[5]); // Taro NP
    const Tree* tree2_4 = APPLY_BINARY(Bx, leaves2[1], leaves2[2]); // saw briefly (S[dcl]\NP)/NP
    const Tree* tree2_5 = APPLY_BINARY(conj, leaves2[4], tree2_3); // and Taro NP\NP
    const Tree* tree2_6 = APPLY_BINARY(bwd, tree2_2, tree2_5); // Tom and Taro NP
    const Tree* tree2_7 = APPLY_BINARY(fwd, tree2_4, tree2_6);
    const Tree* tree2_8 = APPLY_BINARY(bwd, tree2_1, tree2_7);
    const Tree* tree2_9 = APPLY_BINARY(rp, tree2_8, leaves2[6]);

    APPLICABLE(Bx, leaves2[1], leaves2[2]);
    APPLICABLE(conj, leaves2[4], tree2_3);
    APPLICABLE(bwd, tree2_2, tree2_5);
    APPLICABLE(fwd, tree2_4, tree2_6);
    APPLICABLE(bwd, tree2_1, tree2_7);
    APPLICABLE(rp, tree2_8, leaves2[6]);

    print(tree2_9->ToStr());
    ShowDerivation(tree2_9);


}
} // namespace tree

namespace tagger {

using namespace myccg;
#define RANGE(array, i, size) (array) + ((i) * (size)), \
                        (array) + ((i) * (size) + (size) - 1)

void test()
{
    int max_idx;
    std::cout << "----" << __FILE__ << "----" << std::endl;
    const std::string path = "../model";
    tagger::ChainerTagger tagg(path);
    const std::string sent = "this is a new sentence .";
    print(sent);
    std::vector<std::string> tokens = utils::Split(sent, ' ');
    auto res = tagg.predict(sent);
    for (unsigned i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res.get(), i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    const std::string sent2 = "Ed saw briefly his friend .";
    print(sent2);
    std::vector<std::string> tokens2 = utils::Split(sent2, ' ');
    auto res2 = tagg.predict(sent2);
    for (unsigned i = 0; i < tokens2.size(); i++) {
        std::cout << tokens2[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res2.get(), i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    const std::string sent3 = "Darth Vador , also known as Anakin Skywalker is a fictional character .";
    print(sent3);
    std::vector<std::string> tokens3 = utils::Split(sent3, ' ');
    auto res3 = tagg.predict(sent3);
    for (unsigned i = 0; i < tokens3.size(); i++) {
        std::cout << tokens3[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res3.get(), i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    std::cout << "batch experiment" << std::endl;
    std::vector<std::string> doc{sent, sent2, sent3};
    auto res4 = tagg.predict(doc);
    auto res5 = res4[0];
    for (unsigned i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res5, i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }
    auto res6 = res4[1];
    for (unsigned i = 0; i < tokens2.size(); i++) {
        std::cout << tokens2[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res6, i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }

    auto res7 = res4[2];
    for (unsigned i = 0; i < tokens3.size(); i++) {
        std::cout << tokens3[i] << " --> ";
        max_idx = utils::ArgMax(RANGE(res7, i, tagg.TargetSize()));
        std::cout << tagg.TagAt(max_idx)->ToStr() << std::endl;
    }
}
} // namespace tagger
} // namespace myccg
