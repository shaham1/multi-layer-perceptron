#include <iostream>
#include <vector>
#include <concepts>
#include <cassert>
#include <random>
#include <numeric>

template<typename T>
concept NeuralScalar = std::floating_point<T>;

template<NeuralScalar T>
class Matrix {
    private:
        std::vector<T> data;
        size_t rows;
        size_t columns;
    
    public:
        Matrix(size_t rows, size_t columns) 
        : rows(rows), columns(columns) {
        data.resize(rows * columns, static_cast<T>(0));
        }

        [[nodiscard]] T& operator()(size_t r, size_t c) {
            assert(r < rows && c < columns && "Index out of bounds");
            return data[r * columns + c];
        }
        [[nodiscard]] const T& operator()(size_t r, size_t c) const {
            assert(r < rows && c < columns && "Index out of bounds");
            return data[r * columns + c];
        }

        [[nodiscard]] size_t get_rows() const { return rows; }
        [[nodiscard]] size_t get_cols() const { return columns; }

        Matrix operator+=(Matrix& other) {
            assert(rows == other.rows && columns == other.columns && "Matrices do not have the same dimensions.");
            
            for (size_t i=0; i < data.size(); ++i) {
                data[i] += other.data[i];
            }

            return *this;
        }

        Matrix operator-=(Matrix& other) {
            assert(rows == other.rows && columns == other.columns && "Matrices do not have the same dimensions.");

            for (size_t i; i < data.size(); ++i) {
                data[i] -= other.data[i];
            }
            return *this;
        }

        Matrix operator*=(float num) {
            for (size_t i; i < data.size(); ++i) {
                data[i] *= num;
            }
            return *this;
        }

        Matrix dot(const Matrix& other) {
            assert(columns == other.rows && "columns on LHS must be equal to rows on RHS");
            Matrix result(rows, other.columns);

            for (size_t r=0; r < rows; ++r) {
                for (size_t c=0; c < other.columns; ++c) {
                    float total = 0;
                    for (size_t n=0; n < other.columns; ++n) {
                        total += this(r, n) * other(n, r);
                    }
                    result(r, c) = total;
                }
            }
            return result;
        }

        Matrix transpose() const {      
            Matrix result(columns, rows);
            
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < columns; ++c) {
                    result(c, r) = (*this)(r, c);
                }
            }
            return result;
        }

        void populate_random() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(0.0, 1.0);

            for (int i = 0; i < (rows * columns); i++) {
                data[i] = dis(gen);
            }
        }

        void print() const {
            std::cout << "Matrix (" << rows << "x" << columns << "):\n";
            for (size_t r = 0; r < rows; ++r) {
                std::cout << "[ ";
                for (size_t c = 0; c < columns; ++c) {
                    std::cout << (*this)(r, c) << " "; 
                }
                std::cout << "]\n";
            }
            std::cout << "\n";
        }
};

void train();

int main() {

    Matrix<float> input(2, 3);
    input.populate_random();
    input.print();

    Matrix<float> output(3, 3);
    output.populate_random();
    output.print();

    Matrix<float> result(2, 3);
    output += input;
    output.print();

    return 0;
}
