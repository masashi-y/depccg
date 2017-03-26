
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
      use_beta_(true),
      possible_root_cats_(possible_root_cats),
      comparator_(comparator),
      binary_rules_(binary_rules),
      beta_(beta),
      pruning_size_(pruning_size),
      logger_(loglevel) {}

    virtual ~Parser() {};

    virtual void LoadSeenRules() = 0;
    virtual void LoadCategoryDict() = 0;
    void SetComparator(Comparator comp) { comparator_ = comp; }
    void SetBeta(float beta) { beta_ = beta; }
    void SetUseBeta(bool use_beta) { use_beta_ = use_beta; }
    void SetPruningSize(int prune) { pruning_size_ = prune; }
    ParserLogger& GetLogger() { return logger_; }
    NodeType Failed(const std::string& sent, const std::string& message);

    virtual NodeType Parse(int id, const std::string& sent, float* scores) NO_IMPLEMENTATION
    virtual NodeType Parse(int id, const std::string& sent, float* tag_scores, float* dep_scores) NO_IMPLEMENTATION
    virtual std::vector<NodeType> Parse(const std::vector<std::string>& doc) NO_IMPLEMENTATION
    virtual std::vector<NodeType> Parse(const std::vector<std::string>& doc, float** scores) NO_IMPLEMENTATION
    virtual std::vector<NodeType> Parse(const std::vector<std::string>& doc, float** tag_scores, float** dep_scores) NO_IMPLEMENTATION

    // to capture SIGINT or SIGTERM
    static bool keep_going;

protected:
    Tagger* tagger_;
    std::string model_;

    std::unordered_map<Cat, std::vector<Cat>> unary_rules_;

    bool  use_beta_;

    std::unordered_set<Cat> possible_root_cats_;
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
            comparator, binary_rules, beta, pruning_size, loglevel),
      use_seen_rules_(false),
      use_category_dict_(false),
      rule_cache_(binary_rules_) {}

    std::vector<NodeType> Parse(const std::vector<std::string>& doc);
    std::vector<NodeType> Parse(const std::vector<std::string>& doc, float** scores);
    NodeType Parse(int id, const std::string& sent, float* scores);


protected:
    bool IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const;
    bool IsSeen(Cat left, Cat right) const;
    Cat TagAt(int index) const { return tagger_->TagAt(index); }
    int TagSize() const { return tagger_->TargetSize(); }
    std::vector<RuleCache>& GetRules(Cat left, Cat right);

    void LoadSeenRules();
    void LoadCategoryDict();

    template<typename T>
    class SeenRules {
    public:
        SeenRules() {}
        SeenRules(const std::string& file)
        : rules_(utils::LoadSeenRules(file, SeenRules<T>::Preprocess)) {}

        static Cat Preprocess(Cat cat);

        bool IsSeen(Cat left, Cat right) const {
            return (rules_.count(std::make_pair(
                    Preprocess(left), Preprocess(right))) > 0);
        }

    private:
        std::unordered_set<CatPair> rules_;
    };

    template<typename T>
    class CachedRules {
    public:
        CachedRules(std::vector<Op> binary_rules)
        : binary_rules_(binary_rules) {}

        Cat Preprocess(Cat cat);
        
        std::vector<RuleCache>& GetRules(Cat left1, Cat right1) {
            Cat left = Preprocess(left1);
            Cat right = Preprocess(right1);
            auto key = std::make_pair(left, right);
            if (rule_cache_.count(key) > 0)
                return rule_cache_[key];
            std::vector<RuleCache> tmp;
            for (auto rule: binary_rules_) {
                if (rule->CanApply(left, right)) {
                    tmp.emplace_back(rule->Apply(left, right),
                                rule->HeadIsLeft(left, right), rule);
                }
            }
    #pragma omp critical(GetRules)
            rule_cache_.emplace(key, tmp);
            return rule_cache_[key];
        }

    private:
        std::vector<Op> binary_rules_;
        std::unordered_map<CatPair, std::vector<RuleCache>> rule_cache_;
    };

    bool use_seen_rules_;
    SeenRules<Lang> seen_rules_;

    bool use_category_dict_;
    std::unordered_map<std::string, std::vector<bool>> category_dict_;

    CachedRules<Lang> rule_cache_;
};

class En;

template<typename Lang> template<typename T>
Cat AStarParser<Lang>::SeenRules<T>::Preprocess(Cat cat) { return cat; }

template<> template<>
Cat AStarParser<En>::SeenRules<En>::Preprocess(Cat cat);

template<typename Lang> template<typename T>
Cat AStarParser<Lang>::CachedRules<T>::Preprocess(Cat cat) { return cat; }

template<> template<>
Cat AStarParser<En>::CachedRules<En>::Preprocess(Cat cat);
        

} // namespace myccg
#endif
