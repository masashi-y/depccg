
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
    RuleCache(Cat result, bool left_is_head, const combinator::Combinator* combinator)
    : result(result), left_is_head(left_is_head), combinator(combinator) {}

    Cat result;
    bool left_is_head;
    const combinator::Combinator* combinator;
};

class Parser
{
    virtual NodePtr Parse(const std::string& sent, float beta) = 0;

    virtual NodePtr Parse(const std::string& sent, float* scores, float beta) = 0;
};

class AStarParser: public Parser
{
public:
    AStarParser(
            const tagger::Tagger* tagger,
            const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
            const std::vector<combinator::Combinator*>& binary_rules,
            const std::unordered_set<CatPair>& seen_rules,
            const std::unordered_set<Cat>& possible_root_cats)
     :tagger_(tagger),
      unary_rules_(unary_rules),
      binary_rules_(binary_rules),
      seen_rules_(seen_rules),
      possible_root_cats_(possible_root_cats) {}

    NodePtr Parse(const std::string& sent, float beta=0.0000001);

    std::vector<NodePtr> Parse(const std::vector<std::string>& doc, float beta=0.0000001);

    NodePtr Parse(const std::string& sent, float* scores, float beta=0.0000001);

private:

    bool IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const ;

    bool IsSeen(Cat left, Cat right) const;

    Cat TagAt(int index) const { return tagger_->TagAt(index); }

    int TagSize() const { return tagger_->TargetSize(); }

    std::vector<RuleCache>& GetRules(Cat left, Cat right);

    NodePtr failure_node = std::make_shared<tree::Tree>(
            cat::Parse("XX"), new tree::Leaf("FAILURE", cat::Parse("XX"), 0));


    const tagger::Tagger* tagger_;
    std::unordered_map<Cat, std::vector<Cat>> unary_rules_;
    std::vector<combinator::Combinator*> binary_rules_;
    std::unordered_set<CatPair> seen_rules_;
    std::unordered_set<Cat> possible_root_cats_;
    std::unordered_map<CatPair, std::vector<RuleCache>> rule_cache_;
};
        
} // namespace parser
} // namespace myccg
#endif
