#include "matrix_data.hpp"


bool operator==(const Shape& left, const Shape& right) {
    return left.cols == right.cols && left.rows == right.rows;
}

bool operator!=(const Shape& left, const Shape& right) {
    return !(left == right);
}
