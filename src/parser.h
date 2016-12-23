
#ifndef INCLUDE_PARSER_H_
#define INCLUDE_PARSER_H_

#include "tree.h"
#include "chainer_tagger.h"
#include "combinator.h"
#include "parser_tools.h"
#include "cat.h"
#include "cat_loader.h"
#include <iostream>

namespace myccg {

typedef bool (*Comparator)(const AgendaItem&, const AgendaItem&);

class Parser
{
public:
    Parser(const Tagger* tagger,
           const std::string& model,
           const std::unordered_set<Cat>& possible_root_cats,
           Comparator comparator=LongerDependencyComparator,
           float beta=0.0000001,
           int pruning_size=50)
     :tagger_(tagger),
      model_(model),
      unary_rules_(utils::LoadUnary(model + "/unary_rules.txt")),
      use_seen_rules_(false),
      use_category_dict_(false),
      possible_root_cats_(possible_root_cats),
      comparator_(comparator),
      beta_(beta),
      pruning_size_(pruning_size) {}

    virtual ~Parser() {}

    void LoadSeenRules() {
        use_seen_rules_ = true;
        seen_rules_ = utils::LoadSeenRules(model_ + "/seen_rules.txt");
    }

    void LoadCategoryDict() {
        use_category_dict_ = true;
        category_dict_ = utils::LoadCategoryDict(
                model_ + "/cat_dict.txt", tagger_->Targets());
    }

    void SetComparator(Comparator comp) { comparator_ = comp; }
    void SetBeta(float beta) { beta_ = beta; }
    void SetPruningSize(int prune) { pruning_size_ = prune; }

    virtual std::vector<NodeType> Parse(const std::vector<std::string>& doc) = 0;

protected:
    NodeType failure_node = std::make_shared<Tree>(
            Category::Parse("XX"), new Leaf("FAILURE", Category::Parse("XX"), 0));


    const Tagger* tagger_;
    std::string model_;

    std::unordered_map<Cat, std::vector<Cat>> unary_rules_;

    bool use_seen_rules_;
    std::unordered_set<CatPair> seen_rules_;

    bool use_category_dict_;
    std::unordered_map<std::string, std::vector<bool>> category_dict_;

    std::unordered_set<Cat> possible_root_cats_;
    std::unordered_map<CatPair, std::vector<RuleCache>> rule_cache_;
    Comparator comparator_;
    float beta_;
    int pruning_size_;
};

template<typename Lang>
class AStarParser: public Parser
{
public:
    AStarParser(
            const Tagger* tagger,
            const std::string& model,
            const std::unordered_set<Cat>& possible_root_cats,
            Comparator comparator=LongerDependencyComparator,
            float beta=0.0000001,
            int pruning_size=50)
    : Parser(tagger, model, possible_root_cats,
                        comparator, beta, pruning_size) {}

    std::vector<NodeType> Parse(const std::vector<std::string>& doc);
    NodeType Parse( const std::string& sent, float* scores);


protected:
    bool IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const ;
    bool IsSeen(Cat left, Cat right) const;
    Cat TagAt(int index) const { return tagger_->TagAt(index); }
    int TagSize() const { return tagger_->TargetSize(); }
    std::vector<RuleCache>& GetRules(Cat left, Cat right);
};
        
template<typename Lang>
class DepAStarParser: public AStarParser<Lang>
{
public:

    typedef AStarParser<Lang> Base;

    DepAStarParser(
            const Tagger* tagger,
            const std::string& model,
            const std::unordered_set<Cat>& possible_root_cats,
            Comparator comparator=LongerDependencyComparator,
            float beta=0.0000001,
            int pruning_size=50)
    : AStarParser<Lang>(tagger, model, possible_root_cats,
                                comparator, beta, pruning_size) {}

    std::vector<NodeType> Parse(const std::vector<std::string>& doc);
    NodeType Parse(const std::string& sent, float* tag_scores, float* dep_scores);
};


} // namespace myccg
#endif
