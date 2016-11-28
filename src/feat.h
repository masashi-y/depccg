
#include <iostream>


namespace myccg {
namespace cat {

class FeatureValue
{
// private:
    // static int num_feats;

public:
    FeatureValue(const std::string& value): value_(value) {}

    ~FeatureValue() {}

    std::string ToStr() const { return IsEmpty() ? "" : "[" + value_ + "]"; }

    bool operator==(const FeatureValue& other) const { return value_ == other.value_; }

    bool IsEmpty() const { return value_.empty(); }

    bool Matches(const FeatureValue* other) const { return value_ == other->value_; }

private:
    std::string value_;
};

typedef const FeatureValue* Feat;
extern Feat kWILDCARD;
extern Feat kNONE;
extern Feat kNB;

} // namespace cat
} // namespace myccg
