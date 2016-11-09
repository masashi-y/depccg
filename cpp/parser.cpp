
#include "parser.h"
#include <cmath>
#include <unordered_map>
#include <queue>
#include <utility>
#include <limits>
#include <memory>

namespace myccg {
namespace parser {

struct AgendaItem
{
    AgendaItem(NodePtr parse, float in_prob, float out_prob,
            int start_of_span, int span_length)
    : parse(parse), in_prob(in_prob), out_prob(out_prob),
    cost(in_prob + out_prob), start_of_span(start_of_span),
    span_length(span_length) {}

    ~AgendaItem() {}

    NodePtr parse;
    float in_prob;
    float out_prob;
    float cost;
    int start_of_span;
    int span_length;

};
bool operator<(const AgendaItem& item1, const AgendaItem& item2) {
    if ( fabs(item1.cost - item2.cost) > 0.0000001 )
        return item1.cost < item2.cost;
    return item1.parse->GetDependencyLength() > item2.parse->GetDependencyLength();
}

typedef std::pair<NodePtr, float> ChartItem;
class ChartCell
{
public:
    ChartCell():
    items_(std::unordered_map<Cat, ChartItem>()),
    best_cost_(std::numeric_limits<float>::max()), best_(NULL) {}

    bool IsEmpty() const { return items_.size() == 0; }

    NodePtr GetBestParse() const { return best_; }

    bool update(NodePtr parse, float cost) {
        const cat::Category* cat = parse->GetCategory();
        if (items_.count(cat) > 0 && cost > best_cost_)
            return false;
        items_[cat] = std::make_pair(parse, cost);
        if (best_cost_ > cost) {
            best_cost_ = cost;
            best_ = parse;
        }
        return true;
    }

    std::unordered_map<Cat, ChartItem> items_;
private:
    float best_cost_;
    NodePtr best_;
};


AStarParser::AStarParser(const tagger::Tagger* tagger, const std::string& model)
 :tagger_(tagger),
  unary_rules_(utils::load_unary(model + "/unary_rules")),
  binary_rules_(combinator::binary_rules),
  seen_rules_(utils::load_seen_rules(model + "/seenRules")),
  possible_root_cats_({cat::parse("S[dcl]"), cat::parse("S[wq]"),
    cat::parse("S[q]"), cat::parse("S[qem]"), cat::parse("NP")}) {
      rule_cache_.clear();
  }


bool AStarParser::AcceptableRootOrSubtree(Cat cat, int span_len, int s_len) const {
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
    rule_cache_.emplace(key, std::vector<RuleCache>());
    for (auto rule: binary_rules_) {
        if (rule->CanApply(left, right)) {
            rule_cache_[key].emplace_back(
                        left, right, rule->Apply(left, right),
                        rule->HeadIsLeft(left, right), rule);
        }
    }
    return rule_cache_[key];
}

void ComputeOutsideProbs(float* probs, std::size_t sentSize, float* out) {
    int j;
    float* from_left = new float[sentSize + 1];
    float* from_right = new float[sentSize + 1];
    from_left[0] = 0.0;
    from_right[sentSize] = 0.0;

    for (int i = 0; i < sentSize; i++) {
        j = sentSize - i;
        from_left[i + 1] = from_left[i] + probs[i];
        from_right[j - 1] = from_right[j] + probs[j - 1];
    }

    for (int i = 0; i < sentSize + 1; i++) {
        for (int j = 0; j < sentSize + 1; j++) {
            out[i * sentSize + j] = from_left[i] + from_right[j];
        }
    }
    delete[] from_left;
    delete[] from_right;
}

const tree::Node* AStarParser::Parse(const std::string& sent, float beta) {
    std::unique_ptr<float[]> scores = tagger_->predict(sent);
    const tree::Node* res = Parse(sent, scores.get(), beta);
    return res;
}

const tree::Node* AStarParser::Parse(const std::string& sent, float* scores, float beta) {
    std::vector<std::string> tokens = utils::split(sent, ' ');
    std::size_t length = tokens.size();
    std::unique_ptr<float[]> best_in_probs(new float[length + 1]);
    std::unique_ptr<float[]> out_probs(new float[(length + 1) * (length + 1)]);
    std::priority_queue<AgendaItem> agenda;

    std::vector<std::vector<std::pair<float, Cat>>> scored_cats;

    for (int i = 0; i < tokens.size(); i++) {
        scored_cats.emplace_back(std::vector<std::pair<float, Cat>>());
        float total = 0.0;
        best_in_probs[i] = 0.0;
        for (int j = 0; j < TagSize(); j++) {
            float score = scores[i * TagSize() + j];
            total += score;
            if (score >= best_in_probs[i]) best_in_probs[i] = score;
            scored_cats[i].emplace_back(score, TagAt(j));
        }
        std::sort(scored_cats[i].begin(), scored_cats[i].end(),
            [](std::pair<float, Cat>& left, std::pair<float, Cat>& right) {
                return left.first > right.first;});
        float threshold = best_in_probs[i] * beta;
        // normalize and pruning
        for (int j = 0; j < scored_cats[i].size(); j++) {
            if (scored_cats[i][j].first > threshold) 
                scored_cats[i][j].first =
                    std::exp(scored_cats[i][j].first) / std::exp(total);
            else
                scored_cats[i].pop_back();
        }
        best_in_probs[i] = std::exp(best_in_probs[i]) / std::exp(total);
    }
    ComputeOutsideProbs(best_in_probs.get(), length, out_probs.get());

    for (int i = 0; i < scored_cats.size(); i++) {
        for (int j = 0; j < scored_cats[i].size(); j++) {
            auto& prob_and_cat = scored_cats[i][j];
            float out_prob = out_probs[i * length + (i + 1)];
            agenda.emplace(
                    std::make_shared<tree::Leaf>(tokens[i], prob_and_cat.second, i),
                    prob_and_cat.first, out_prob, i, 1);
        }
    }

    ChartCell* chart = new ChartCell[length * length];
    for (int i = 0; i < (length * length); i++) {
        chart[i] = ChartCell();
    }

    while (chart[length - 1].IsEmpty() && agenda.size() > 0) {
        const AgendaItem& item = agenda.top();
        NodePtr parse = item.parse;
        ChartCell& cell = chart[item.start_of_span * length + (item.span_length - 1)];

        if (cell.update(parse, item.in_prob)) {
            if (item.span_length != length) {
                for (Cat unary: unary_rules_[parse->GetCategory()]) {
                    NodePtr subtree = std::make_shared<const tree::Tree>(unary, true, parse, new combinator::UnaryRule());
                    float out_prob = out_probs[item.start_of_span * length +
                                    item.start_of_span + item.span_length];
                    agenda.push(AgendaItem(subtree,
                                        item.in_prob,
                                        out_prob,
                                        item.start_of_span,
                                        item.span_length));
                }
            }

            for (int span_length = item.span_length + 1
                ; span_length < 1 + length - item.start_of_span
                ; span_length++) {
                ChartCell& other = chart[(item.start_of_span + item.span_length) * 
                                length + (span_length - item.span_length - 1)];
                for (auto&& pair: other.items_) {
                    NodePtr right = pair.second.first;
                    float prob = pair.second.second;
                    for (auto&& rule: GetRules(
                                parse->GetCategory(), right->GetCategory())) {
                        if (IsNormalForm(rule.combinator->GetRuleType(), parse, right) &&
                                AcceptableRootOrSubtree(rule.result, span_length, length)) {
                            NodePtr subtree = std::make_shared<const tree::Tree>(rule.result, rule.left_is_head, parse, right, rule.combinator);
                            float in_prob = item.in_prob + prob;
                            float out_prob = out_probs[item.start_of_span * length +
                                item.start_of_span + span_length];
                            agenda.push(AgendaItem(subtree,
                                                in_prob,
                                                out_prob,
                                                item.start_of_span,
                                                span_length));
                        }
                    }
                }
            }
            for (int start_of_span = 0; start_of_span < item.start_of_span; start_of_span++) {
                int span_length = item.start_of_span + item.span_length - start_of_span;
                ChartCell& other = chart[start_of_span * length +
                                    (span_length - item.span_length - 1)];
                for (auto&& pair: other.items_) {
                    NodePtr left = pair.second.first;
                    float prob = pair.second.second;
                    for (auto&& rule: GetRules(
                                left->GetCategory(), parse->GetCategory())) {
                        if (IsNormalForm(rule.combinator->GetRuleType(), left, parse) &&
                                AcceptableRootOrSubtree(rule.result, span_length, length)) {
                            NodePtr subtree = std::make_shared<const tree::Tree>(rule.result, rule.left_is_head, left, parse, rule.combinator);
                            float in_prob = item.in_prob + prob;
                            float out_prob = out_probs[start_of_span * length +
                                            start_of_span + span_length];
                            agenda.push(AgendaItem(subtree,
                                                in_prob,
                                                out_prob,
                                                start_of_span,
                                                span_length));
                        }
                    }
                }
            }
        }
        agenda.pop();
    }
    if (chart[length - 1].IsEmpty())
        return failure_node;
    return chart[length - 1].GetBestParse().get();
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

}

} // namespace parser
} // namespace myccg

int main()
{
    // myccg::tagger::test();
    myccg::tree::test();
    myccg::utils::test();
    myccg::combinator::test();
    myccg::parser::test();
}

