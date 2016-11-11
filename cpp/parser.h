
#ifndef INCLUDE_PARSER_H_
#define INCLUDE_PARSER_H_

#include "tree.h"
#include "chainer_tagger.h"
#include "combinator.h"

namespace myccg {
namespace parser {

using cat::Cat;
using cat::CatPair;
typedef std::shared_ptr<const tree::Node> NodePtr;


struct RuleCache
{
    RuleCache(Cat lchild, Cat rchild, Cat result, bool left_is_head,
            const combinator::Combinator* combinator)
    : lchild(lchild), rchild(rchild), result(result),
      left_is_head(left_is_head), combinator(combinator) {}

    Cat lchild;
    Cat rchild;
    Cat result;
    bool left_is_head;
    const combinator::Combinator* combinator;
};

class Parser
{
    virtual const tree::Tree* Parse(const std::string& sent, float beta) = 0;

    virtual const tree::Tree* Parse(const std::string& sent, float* scores, float beta) = 0;
};

class AStarParser: public Parser
{
public:
    AStarParser(const tagger::Tagger* tagger, const std::string& model);

    const tree::Tree* Parse(const std::string& sent, float beta=0.0000001);

    std::vector<const tree::Tree*> Parse(const std::vector<std::string>& doc, float beta=0.0000001);

    const tree::Tree* Parse(const std::string& sent, float* scores, float beta=0.0000001);

    void test();
private:

    bool IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const ;

    bool IsSeen(Cat left, Cat right) const;

    Cat TagAt(int index) const { return tagger_->TagAt(index); }

    int TagSize() const { return tagger_->TargetSize(); }

    std::vector<RuleCache>& GetRules(Cat left, Cat right);

    const tree::Tree* failure_node = new tree::Tree(
            cat::parse("XX"), new tree::Leaf("FAILURE", cat::parse("XX"), 0));


    const tagger::Tagger* tagger_;
    std::unordered_map<Cat, std::vector<Cat>> unary_rules_;
    std::vector<combinator::Combinator*> binary_rules_;
    std::unordered_set<CatPair, utils::hash_cat_pair> seen_rules_;
    std::vector<Cat> possible_root_cats_;
    std::unordered_map<CatPair, std::vector<RuleCache>, utils::hash_cat_pair> rule_cache_;
};
        
void test();
} // namespace parser
} // namespace myccg
#endif
