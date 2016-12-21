
#ifndef INCLUDE_PARSER_H_
#define INCLUDE_PARSER_H_

#include "tree.h"
#include "chainer_tagger.h"
#include "combinator.h"
#include "parser_tools.h"
#include "cat.h"

namespace myccg {

typedef bool (*Comparator)(const AgendaItem&, const AgendaItem&);

template<typename Lang>
class AStarParser
{
public:
    AStarParser(
            const Tagger* tagger,
            const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
            const std::unordered_set<CatPair>& seen_rules,
            const std::unordered_set<Cat>& possible_root_cats,
            Comparator comparator,
            float beta=0.0000001,
            int pruning_size=50)
     :tagger_(tagger),
      unary_rules_(unary_rules),
      seen_rules_(seen_rules),
      possible_root_cats_(possible_root_cats),
      comparator_(comparator),
      beta_(beta),
      pruning_size_(pruning_size) {}

    virtual NodeType Parse(const std::string& sent);
    virtual std::vector<NodeType> Parse(const std::vector<std::string>& doc);
    NodeType Parse( const std::string& sent, float* scores);

protected:
    bool IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const ;
    bool IsSeen(Cat left, Cat right) const;
    virtual Cat TagAt(int index) const { return tagger_->TagAt(index); }
    virtual int TagSize() const { return tagger_->TargetSize(); }
    std::vector<RuleCache>& GetRules(Cat left, Cat right);

protected:
    NodeType failure_node = std::make_shared<Tree>(
            Category::Parse("XX"), new Leaf("FAILURE", Category::Parse("XX"), 0));

    const Tagger* tagger_;
    std::unordered_map<Cat, std::vector<Cat>> unary_rules_;
    std::unordered_set<CatPair> seen_rules_;
    std::unordered_set<Cat> possible_root_cats_;
    std::unordered_map<CatPair, std::vector<RuleCache>> rule_cache_;
    Comparator comparator_;
    float beta_;
    int pruning_size_;

};
        
template<typename Lang>
class DepAStarParser: public AStarParser<Lang>
{
public:

    typedef AStarParser<Lang> Base;

    DepAStarParser(
            const DependencyTagger* tagger,
            const std::unordered_map<Cat, std::vector<Cat>>& unary_rules,
            const std::unordered_set<CatPair>& seen_rules,
            const std::unordered_set<Cat>& possible_root_cats,
            Comparator comparator)
    : AStarParser<Lang>(NULL, unary_rules, seen_rules,
            possible_root_cats, comparator), dep_tagger_(tagger) {}

    NodeType Parse(const std::string& sent);
    std::vector<NodeType> Parse(const std::vector<std::string>& doc);
    int TagSize() const { return dep_tagger_->TargetSize(); }
    Cat TagAt(int index) const { return dep_tagger_->TagAt(index); }
    NodeType Parse(const std::string& sent, float* tag_scores, float* dep_scores);

private:
    const DependencyTagger* dep_tagger_;
};


} // namespace myccg
#endif
