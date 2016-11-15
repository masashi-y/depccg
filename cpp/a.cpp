#include <iostream>

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


private:
    std::string value_;
};

typedef const FeatureValue* Feat;
Feat kWILDCARD  = new FeatureValue("X");
Feat kNONE      = new FeatureValue("");
Feat kNB        = new FeatureValue("nb");
Feat feat = new FeatureValue("this is test");
int main() {
    std::cout << (new FeatureValue("test"))->ToStr() << std::endl;
    std::cout << (feat)->ToStr() << std::endl;
    std::cout << kNB->ToStr() << std::endl;
}
