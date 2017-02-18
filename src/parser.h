
#ifndef INCLUDE_PARSER_H_
#define INCLUDE_PARSER_H_

#include <iostream>
#include "tree.h"
#include "chainer_tagger.h"
#include "combinator.h"
#include "parser_tools.h"
#include "cat.h"
#include "cat_loader.h"
#include "logger.h"

namespace myccg {

typedef bool (*Comparator)(const AgendaItem&, const AgendaItem&);

class Parser
{
public:
    Parser(Tagger* tagger,
           const std::string& model,
           const std::unordered_set<Cat>& possible_root_cats,
           Comparator comparator,
           std::vector<Op> binary_rules,
           float beta,
           int pruning_size,
           LogLevel loglevel)
     :tagger_(tagger),
      model_(model),
      unary_rules_(utils::LoadUnary(model + "/unary_rules.txt")),
      use_seen_rules_(false),
      use_category_dict_(false),
      use_beta_(true),
      possible_root_cats_(possible_root_cats),
      comparator_(comparator),
      binary_rules_(binary_rules),
      beta_(beta),
      pruning_size_(pruning_size),
      logger_(loglevel) {}

    virtual ~Parser() {}

    void LoadSeenRules();
    void LoadCategoryDict();
    void SetComparator(Comparator comp) { comparator_ = comp; }
    void SetBeta(float beta) { beta_ = beta; }
    void SetUseBeta(bool use_beta) { use_beta_ = use_beta; }
    void SetPruningSize(int prune) { pruning_size_ = prune; }

    virtual std::vector<NodeType> Parse(const std::vector<std::string>& doc) = 0;

    static bool keep_going;

protected:
    NodeType failure_node = std::make_shared<Leaf>("fail", Category::Parse("NP"), 0);


    Tagger* tagger_;
    std::string model_;

    std::unordered_map<Cat, std::vector<Cat>> unary_rules_;

    bool use_seen_rules_;
    std::unordered_set<CatPair> seen_rules_;

    bool use_category_dict_;
    std::unordered_map<std::string, std::vector<bool>> category_dict_;

    bool  use_beta_;

    std::unordered_set<Cat> possible_root_cats_;
    std::unordered_map<CatPair, std::vector<RuleCache>> rule_cache_;
    Comparator comparator_;
    std::vector<Op> binary_rules_;
    float beta_;
    int pruning_size_;
    ParserLogger logger_;
};

template<typename Lang>
class AStarParser: public Parser
{
public:
    AStarParser(
            Tagger* tagger,
            const std::string& model,
            const std::unordered_set<Cat>& possible_root_cats,
            Comparator comparator,
            std::vector<Op> binary_rules,
            float beta,
            int pruning_size,
            LogLevel loglevel)
    : Parser(tagger, model, possible_root_cats,
            comparator, binary_rules, beta, pruning_size, loglevel) {}

    std::vector<NodeType> Parse(const std::vector<std::string>& doc);
    NodeType Parse(int id, const std::string& sent, float* scores);


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
            Tagger* tagger,
            const std::string& model,
            const std::unordered_set<Cat>& possible_root_cats,
            Comparator comparator,
           std::vector<Op> binary_rules,
            float beta,
            int pruning_size,
            LogLevel loglevel)
    : AStarParser<Lang>(tagger, model, possible_root_cats,
                comparator, binary_rules, beta, pruning_size, loglevel) {}

    std::vector<NodeType> Parse(const std::vector<std::string>& doc);

    NodeType Parse(int id, const std::string& sent, float* tag_scores, float* dep_scores);
};


} // namespace myccg
#endif
