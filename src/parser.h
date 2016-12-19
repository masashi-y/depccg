
#ifndef INCLUDE_PARSER_H_
#define INCLUDE_PARSER_H_

#include "tree.h"
#include "chainer_tagger.h"
#include "combinator.h"
#include "cat.h"

namespace myccg {
namespace parser {

using cat::Cat;
using cat::CatPair;

typedef std::shared_ptr<const tree::Node> NodeType;


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
    virtual NodeType Parse(const std::string& sent, float beta) = 0;

    virtual std::vector<NodeType>
    Parse(const std::vector<std::string>& doc, float beta) = 0;
    // virtual NodeType Parse(const std::string& sent, float* scores, float beta) = 0;
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

    virtual NodeType Parse(const std::string& sent, float beta=0.0000001);

    virtual std::vector<NodeType> Parse(const std::vector<std::string>& doc, float beta=0.0000001);

    NodeType Parse( const std::string& sent, float* scores, float beta=0.0000001);
protected:

    bool IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const ;

    bool IsSeen(Cat left, Cat right) const;

    virtual Cat TagAt(int index) const { return tagger_->TagAt(index); }

    virtual int TagSize() const { return tagger_->TargetSize(); }

    std::vector<RuleCache>& GetRules(Cat left, Cat right);

    NodeType failure_node = std::make_shared<tree::Tree>(
            cat::Category::Parse("XX"), new tree::Leaf("FAILURE", cat::Category::Parse("XX"), 0));


protected:
    const tagger::Tagger* tagger_;
    std::unordered_map<Cat, std::vector<Cat>> unary_rules_;
    std::vector<combinator::Combinator*> binary_rules_;
    std::unordered_set<CatPair> seen_rules_;
    std::unordered_set<Cat> possible_root_cats_;
    std::unordered_map<CatPair, std::vector<RuleCache>> rule_cache_;
};
        
class DepAStarParser: public AStarParser
{
public:
    DepAStarParser(
            const tagger::DependencyTagger* tagger,
            const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
            const std::vector<combinator::Combinator*>& binary_rules,
            const std::unordered_set<CatPair>& seen_rules,
            const std::unordered_set<Cat>& possible_root_cats)
    : AStarParser(NULL, unary_rules,
            binary_rules, seen_rules, possible_root_cats), dep_tagger_(tagger) {}

    NodeType Parse(const std::string& sent, float beta=0.0000001);

    std::vector<NodeType> Parse(const std::vector<std::string>& doc, float beta=0.0000001);
    int TagSize() const { return dep_tagger_->TargetSize(); }

    Cat TagAt(int index) const { return dep_tagger_->TagAt(index); }

    NodeType Parse(const std::string& sent, float* tag_scores, float* dep_scores, float beta=0.0000001);

private:
    const tagger::DependencyTagger* dep_tagger_;
};


void ComputeOutsideProbs(float* probs, int sent_size, float* out);
} // namespace parser
} // namespace myccg
#endif
