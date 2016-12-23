
#include <limits>
#include <cmath>
#include "parser_tools.h"
#include "configure.h"
#include "grammar.h"

namespace myccg {

bool JapaneseComparator(const AgendaItem& left, const AgendaItem& right) {
    if ( fabs(left.prob - right.prob) > 0.00001 )
        return left.prob < right.prob;

    // if ((IsVerb(left.parse) || IsAdjective(left.parse->GetCategory())) &&
    //         !(IsVerb(right.parse) || IsAdjective(right.parse->GetCategory())))
    //     return false;
    // if ((IsVerb(right.parse) || IsAdjective(right.parse->GetCategory())) &&
    //         !(IsVerb(left.parse) || IsAdjective(left.parse->GetCategory())))
    //     return true;
    // if (IsPeriod(right.parse->GetCategory()))
    //     return false;
    // if (IsPeriod(left.parse->GetCategory()))
    //     return true;
    // if (left.parse->LeftNumDescendants() != right.parse->LeftNumDescendants())
        // return left.parse->LeftNumDescendants() <= right.parse->LeftNumDescendants();
// #else
    if (left.parse->GetDependencyLength() != right.parse->GetDependencyLength())
        return left.parse->GetDependencyLength() < right.parse->GetDependencyLength();

    return left.id > right.id;
}


bool LongerDependencyComparator(const AgendaItem& left, const AgendaItem& right) {
    if ( fabs(left.prob - right.prob) > 0.00001 )
        return left.prob < right.prob;
    if (left.parse->GetDependencyLength() != right.parse->GetDependencyLength())
        return left.parse->GetDependencyLength() > right.parse->GetDependencyLength();
    return left.id > right.id;
}

ChartCell::ChartCell()
    : items(std::unordered_map<Cat, ChartItem>()),
      best_prob(std::numeric_limits<float>::lowest()), best(NULL) {}


bool ChartCell::update(NodeType parse, float prob) {
    Cat cat = parse->GetCategory();
    if (items.count(cat) > 0)
        return false;
    items.emplace(cat, std::make_pair(parse, prob));
    if (best_prob < prob) {
        best_prob = prob;
        best = parse;
    }
    return true;
}

void ComputeOutsideProbs(float* probs, int sent_size, float* out) {
    float from_left[MAX_LENGTH + 1];
    float from_right[MAX_LENGTH + 1];
    from_left[0] = 0.0;
    from_right[sent_size] = 0.0;

    for (int i = 0; i < sent_size - 1; i++) {
        int j = sent_size - i;
        from_left[i + 1] = from_left[i] + probs[i];
        from_right[j - 1] = from_right[j] + probs[j - 1];
    }

    for (int i = 0; i < sent_size + 1; i++) {
        for (int j = i; j < sent_size + 1; j++) {
            out[i * (sent_size + 1) + j] = from_left[i] + from_right[j];
        }
    }
}

}
