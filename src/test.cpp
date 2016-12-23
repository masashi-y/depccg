
#include "matrix.h"
#include "cat.h"
#include <iostream>

using namespace myccg;

int main(int argc, char const* argv[])
{
    int a[10] = {0,6,2,3,4,5,6,7,8,9};
    Matrix<int> m(a, a+10);
    std::cout << m.Size() << std::endl;
    std::cout << m.Column() << std::endl;
    std::cout << m.Row() << std::endl;
    std::cout << Matrix<int>(a, a+5).Size() << std::endl;
    std::cout << Matrix<int>(a, a+5).Reshaped(5, 1).ArgMax() << std::endl;
    std::cout << Matrix<int>(a, 2, 5).ArgMax(1) << std::endl;
    Matrix<float> n(10, 20);
    std::cout << n.Size() << std::endl;
    std::cout << n << std::endl;







    Cat tes = Category::Parse(",");
    std::cout << (tes->IsPunct() ? "yes" : "no") << std::endl;
    return 0;
}
