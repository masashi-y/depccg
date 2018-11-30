
#include <cmath>
#include "parser_tools.h"
#include "grammar.h"

namespace myccg {

const unsigned MAX_LENGTH = 250;

bool JapaneseComparator(const AgendaItem& left, const AgendaItem& right) {
    if ( fabs(left.prob - right.prob) > 0.00001 )
        return left.prob < right.prob;

    if ((Ja::IsVerb(left.parse) || Ja::IsAdjective(left.parse->GetCategory())) &&
            !(Ja::IsVerb(right.parse) || Ja::IsAdjective(right.parse->GetCategory())))
        return false;
    if ((Ja::IsVerb(right.parse) || Ja::IsAdjective(right.parse->GetCategory())) &&
            !(Ja::IsVerb(left.parse) || Ja::IsAdjective(left.parse->GetCategory())))
        return true;
    if (Ja::IsPeriod(right.parse->GetCategory()))
        return false;
    if (Ja::IsPeriod(left.parse->GetCategory()))
        return true;
    if (left.parse->LeftNumDescendants() != right.parse->LeftNumDescendants())
        return left.parse->LeftNumDescendants() <= right.parse->LeftNumDescendants();
    if (left.parse->GetDependencyLength() != right.parse->GetDependencyLength())
        return left.parse->GetDependencyLength() < right.parse->GetDependencyLength();

    return left.id > right.id;
}

bool NormalComparator(const AgendaItem& left, const AgendaItem& right) {
    return left.prob < right.prob;
}


bool LongerDependencyComparator(const AgendaItem& left, const AgendaItem& right) {
    if ( fabs(left.prob - right.prob) > 0.00001 )
        return left.prob < right.prob;
    if (left.parse->GetDependencyLength() != right.parse->GetDependencyLength())
        return left.parse->GetDependencyLength() > right.parse->GetDependencyLength();
    return left.id > right.id;
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
