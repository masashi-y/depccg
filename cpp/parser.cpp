
#include "parser.h"
#include <cmath>
#include <unordered_map>
#include <queue>
#include <utility>
#include <limits>
#include <memory>
#include <omp.h>


// #define DEBUG(var) std::cout << #var": " << (var) << std::endl;

namespace myccg {
namespace parser {

struct AgendaItem
{
    AgendaItem(NodePtr parse_, float in_prob_, float out_prob_,
            int start_of_span_, int span_length_)
    : parse(parse_), in_prob(in_prob_), out_prob(out_prob_),
    prob(in_prob_ + out_prob_), start_of_span(start_of_span_),
    span_length(span_length_) {}

    ~AgendaItem() {}

    NodePtr parse;
    float in_prob;
    float out_prob;
    float prob;
    int start_of_span;
    int span_length;

};
bool operator<(const AgendaItem& item1, const AgendaItem& item2) {
    if ( fabs(item1.prob - item2.prob) > 0.0000001 )
        return item1.prob < item2.prob;
    return item1.parse->GetDependencyLength() > item2.parse->GetDependencyLength();
}

typedef std::pair<NodePtr, float> ChartItem;
class ChartCell
{
public:
    ChartCell():
    items_(std::unordered_map<Cat, ChartItem>()),
    best_prob_(std::numeric_limits<float>::lowest()), best_(NULL) {}

    ~ChartCell() {}

    bool IsEmpty() const { return items_.size() == 0; }

    NodePtr GetBestParse() const { return best_; }

    bool update(NodePtr parse, float prob) {
        Cat cat = parse->GetCategory();
        if (items_.count(cat) > 0 && prob <= best_prob_)
            return false;
        items_.emplace(cat, std::make_pair(parse, prob));;
        if (best_prob_ <= prob) {
            best_prob_ = prob;
            best_ = parse;
        }
        return true;
    }

    std::unordered_map<Cat, ChartItem> items_;
private:
    float best_prob_;
    NodePtr best_;
};


AStarParser::AStarParser(const tagger::Tagger* tagger, const std::string& model)
 :tagger_(tagger),
  unary_rules_(utils::load_unary(model + "/unary_rules.txt")),
  binary_rules_(combinator::binary_rules),
  seen_rules_(utils::load_seen_rules(model + "/seen_rules.txt")),
  possible_root_cats_({cat::parse("S[dcl]"), cat::parse("S[wq]"),
    cat::parse("S[q]"), cat::parse("S[qem]"), cat::parse("NP")}) {}


bool AStarParser::IsAcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const {
    if (span_len == s_len) {
        for (Cat root: possible_root_cats_)
            if (*root == *cat) return true;
        return false;
    }
    return true;
}

bool AStarParser::IsSeen(Cat left, Cat right) const {
    for (auto& seen_rule: seen_rules_) {
        if (*seen_rule.first == *left &&
                *seen_rule.second == *right)
            return true;
    }
    return false;
}

bool IsNormalForm(combinator::RuleType rule_type,
        NodePtr left, NodePtr right) {
    if ( (left->GetRuleType() == combinator::FC ||
                left->GetRuleType() == combinator::GFC) &&
            (rule_type == combinator::FA ||
             rule_type == combinator::FC ||
             rule_type == combinator::GFC) )
        return false;
    if ( (right->GetRuleType() == combinator::BX ||
                left->GetRuleType() == combinator::GBX) &&
            (rule_type == combinator::BA ||
             rule_type == combinator::BX ||
             left->GetRuleType() == combinator::GBX) )
        return false;
    if ( left->GetRuleType() == combinator::UNARY &&
            rule_type == combinator::FA &&
            left->GetCategory()->IsForwardTypeRaised() )
        return false;
    if ( right->GetRuleType() == combinator::UNARY &&
            rule_type == combinator::BA &&
            right->GetCategory()->IsBackwardTypeRaised() )
        return false;
    return true;
}

std::vector<RuleCache>& AStarParser::GetRules(Cat left, Cat right) {
    auto key = std::make_pair(left, right);
    if (rule_cache_.count(key) > 0)
        return rule_cache_[key];
    std::vector<RuleCache> tmp;
    for (auto rule: binary_rules_) {
        if (rule->CanApply(left, right)) {
            tmp.emplace_back(
                        left, right, rule->Apply(left, right),
                        rule->HeadIsLeft(left, right), rule);
        }
    }
    #pragma omp critical
    rule_cache_.emplace(key, tmp);
    return rule_cache_[key];
}

void ComputeOutsideProbs(float* probs, int sent_size, float* out) {
    float* from_left = new float[sent_size + 1];
    float* from_right = new float[sent_size + 1];
    from_left[0] = 0.0;
    from_right[sent_size] = 0.0;

    for (int i = 0; i < sent_size - 1; i++) {
        int j = sent_size - i;
        from_left[i + 1] = from_left[i] + probs[i];
        from_right[j - 1] = from_right[j] + probs[j - 1];
    }

    for (int i = 0; i < sent_size + 1; i++) {
        for (int j = i; j < sent_size + 1; j++) {
            out[i * sent_size + j] = from_left[i] + from_right[j];
        }
    }
    delete[] from_left;
    delete[] from_right;
}

const tree::Tree* AStarParser::Parse(const std::string& sent, float beta) {
    std::unique_ptr<float[]> scores = tagger_->predict(sent);
    const tree::Tree* res = Parse(sent, scores.get(), beta);
    return res;
}

std::vector<const tree::Tree*>
AStarParser::Parse(const std::vector<std::string>& doc, float beta) {
    std::unique_ptr<float*[]> scores = tagger_->predict(doc);
    std::vector<const tree::Tree*> res(doc.size());
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < doc.size(); i++) {
        res[i] = Parse(doc[i], scores[i], beta);
        std::cout << "done: " << i << " length: " << doc[i].size() << std::endl;
    }
    return res;
}

const tree::Tree* AStarParser::Parse(const std::string& sent, float* scores, float beta) {
    int pruning_size = 50;
    std::vector<std::string> tokens = utils::split(sent, ' ');
    int sent_size = (int)tokens.size();
    std::unique_ptr<float[]> best_in_probs(new float[sent_size]);
    std::unique_ptr<float[]> out_probs(new float[(sent_size + 1) * (sent_size + 1)]);
    std::priority_queue<AgendaItem> agenda;

    std::vector<std::vector<std::pair<float, Cat>>> scored_cats;

    for (int i = 0; i < sent_size; i++) {
        scored_cats.emplace_back(std::vector<std::pair<float, Cat>>());
        float total = 0.0;
        for (int j = 0; j < TagSize(); j++) {
            float score = scores[i * TagSize() + j];
            total += std::exp(score);
            scored_cats[i].emplace_back(score, TagAt(j));
        }
        std::sort(scored_cats[i].begin(), scored_cats[i].end(),
            [](std::pair<float, Cat>& left, std::pair<float, Cat>& right) {
                return left.first > right.first;});
        float threshold = scored_cats[i][0].first * beta;
        // normalize and pruning
        for (int j = 0; j < TagSize(); j++) {
            if (scored_cats[i][j].first > threshold) 
                scored_cats[i][j].first =
                    std::log( std::exp(scored_cats[i][j].first) / total );
            else
                scored_cats[i].pop_back();
        }
        best_in_probs[i] = scored_cats[i][0].first;
    }
    ComputeOutsideProbs(best_in_probs.get(), sent_size, out_probs.get());

    for (int i = 0; i < (int)scored_cats.size(); i++) {
        float out_prob = out_probs[i * sent_size + (i + 1)];
        for (int j = 0; j < pruning_size; j++) {
            auto& prob_and_cat = scored_cats[i][j];
            agenda.emplace(
                    std::make_shared<tree::Leaf>(tokens[i], prob_and_cat.second, i),
                    prob_and_cat.first, out_prob, i, 1);
        }
    }

    ChartCell* chart = new ChartCell[sent_size * sent_size];
    // std::unique_ptr<ChartCell[]> chart(new ChartCell[sent_size * sent_size]);

    // int step = 0;
    while (chart[sent_size - 1].IsEmpty() && agenda.size() > 0) {
        const AgendaItem item = agenda.top();
        agenda.pop();
        NodePtr parse = item.parse;
        ChartCell& cell = chart[item.start_of_span * sent_size + (item.span_length - 1)];
        // if (step++ > 20000) break;

        if (cell.update(parse, item.in_prob)) {
            if (item.span_length != sent_size) {
                for (Cat unary: unary_rules_[parse->GetCategory()]) {
                    NodePtr subtree = std::make_shared<const tree::Tree>(unary, parse);
                    agenda.emplace(subtree, item.in_prob, item.out_prob,
                                        item.start_of_span, item.span_length);
                }
            }
            for (int span_length = item.span_length + 1
                ; span_length < 1 + sent_size - item.start_of_span
                ; span_length++) {
                ChartCell& other = chart[(item.start_of_span + item.span_length) * 
                                sent_size + (span_length - item.span_length - 1)];
                for (auto&& pair: other.items_) {
                    NodePtr right = pair.second.first;
                    float prob = pair.second.second;
                    if (! IsSeen(parse->GetCategory(), right->GetCategory())) continue;
                    for (auto&& rule: GetRules(parse->GetCategory(), right->GetCategory())) {
                        if (IsNormalForm(rule.combinator->GetRuleType(), parse, right) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodePtr subtree = std::make_shared<const tree::Tree>(
                                    rule.result, rule.left_is_head, parse, right, rule.combinator);
                            float in_prob = item.in_prob + prob;
                            float out_prob = out_probs[item.start_of_span *
                                            sent_size + item.start_of_span + span_length];
                            agenda.emplace(subtree, in_prob, out_prob,
                                                item.start_of_span, span_length);
                        }
                    }
                }
            }
            for (int start_of_span = 0; start_of_span < item.start_of_span; start_of_span++) {
                int span_length = item.start_of_span + item.span_length - start_of_span;
                ChartCell& other = chart[start_of_span * sent_size +
                                    (span_length - item.span_length - 1)];
                for (auto&& pair: other.items_) {
                    NodePtr left = pair.second.first;
                    float prob = pair.second.second;
                    if (! IsSeen(left->GetCategory(), parse->GetCategory())) continue;
                    for (auto&& rule: GetRules(left->GetCategory(), parse->GetCategory())) {
                        if (IsNormalForm(rule.combinator->GetRuleType(), left, parse) &&
                                IsAcceptableRootOrSubtree(rule.result, span_length, sent_size)) {
                            NodePtr subtree = std::make_shared<const tree::Tree>(
                                    rule.result, rule.left_is_head, left, parse, rule.combinator);
                            float in_prob = item.in_prob + prob;
                            float out_prob = out_probs[start_of_span * sent_size +
                                            start_of_span + span_length];
                            agenda.emplace(subtree, in_prob, out_prob,
                                                start_of_span, span_length);
                        }
                    }
                }
            }
        }
    }
    if (chart[sent_size - 1].IsEmpty())
        return failure_node;
    auto res = chart[sent_size - 1].GetBestParse().get();
    return static_cast<const tree::Tree*>(res);
}

void AStarParser::test() {
    NodePtr leaves[] = {
        NodePtr(new tree::Leaf("this",     cat::parse("NP"),              0)),
        NodePtr(new tree::Leaf("is",       cat::parse("(S[dcl]\\NP)/NP"), 1)),
        NodePtr(new tree::Leaf("a",        cat::parse("NP[nb]/N"),        2)),
        NodePtr(new tree::Leaf("new",      cat::parse("N/N"),             3)),
        NodePtr(new tree::Leaf("sentence", cat::parse("N"),               4)),
        NodePtr(new tree::Leaf(".",        cat::parse("."),               5)),
    };
    print (failure_node->ToStr());
}

void test() {
    std::cout << "----" << __FILE__ << "----" << std::endl;

    const std::string model = "/home/masashi-y/myccg/myccg/model";
    tagger::ChainerTagger tagger(model);
    AStarParser parser(&tagger, model);
    parser.test();
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

    // std::vector<std::string> doc{sent1, sent2, sent3};
    std::vector<std::string> inputs;
    std::string in;
    while (getline(std::cin, in)) {
        inputs.push_back(in);
    }
    sort(inputs.begin(), inputs.end(),
            [](const std::string& s1, const std::string& s2) {
            return s1.size() < s2.size(); });
    auto res_doc = parser.Parse(inputs, 0.0001);
    for (auto&& tree: res_doc) {
        tree::ShowDerivation(tree);
    }

}

} // namespace parser
} // namespace myccg

