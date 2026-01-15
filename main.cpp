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

        Matrix(const std::vector<T>& input_data) 
        : rows(1), columns(input_data.size()) {    
        data = input_data; 
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
        
        void map(float (*func)(float)) {
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = func(data[i]);
            }
        }

        Matrix operator+=(const Matrix& other) {
            assert(rows == other.rows && columns == other.columns && "Matrices do not have the same dimensions.");
            
            for (size_t i=0; i < data.size(); ++i) {
                data[i] += other.data[i];
            }

            return *this;
        }

        Matrix operator-=(const Matrix& other) {
            assert(rows == other.rows && columns == other.columns && "Matrices do not have the same dimensions.");

            for (size_t i; i < data.size(); ++i) {
                data[i] -= other.data[i];
            }
            return *this;
        }

        Matrix operator*=(const float num) {
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
                        total += (*this)(r, n) * other(n, r);
                    }
                    result(r, c) = total;
                }
            }
            return result;
        }
        Matrix multiply(const Matrix& other) const {
            assert(rows == other.rows && columns == other.columns && "Dimensions must match for element-wise multiplication");
            
            Matrix result(rows, columns);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] * other.data[i];
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

struct NetworkResults {
    Matrix<float> hidden_output;
    Matrix<float> output;
};

struct TrainMetrics {
    float error;
    Matrix<float> prediction;
};

class NeuralNetwork {
    private:
        int input_nodes;
        int hidden_nodes;
        int output_nodes;
        float learning_rate;

        Matrix<float> weights_input_hidden;
        Matrix<float> weights_hidden_output;
        Matrix<float> bias_hidden;
        Matrix<float> bias_output;
    
    public:
        NeuralNetwork(int inputs, int hidden, int outputs, float lr):
            input_nodes(inputs),
            hidden_nodes(hidden),
            output_nodes(outputs),
            learning_rate(lr),
            weights_input_hidden(inputs, hidden),
            weights_hidden_output(hidden, outputs),
            bias_hidden(1, hidden),
            bias_output(1, outputs)
        {
            weights_input_hidden.populate_random();
            weights_hidden_output.populate_random();
            bias_hidden.populate_random();
            bias_output.populate_random();
        }
   
        static float sigmoid(float x) {
            return 1.0f / (1.0f + std::exp(-x));
        }
        static float dsigmoid(float y) {
            return y * (1.0f - y);
        }

        NetworkResults predict(const std::vector<float>& input_array) {
            Matrix<float> input(input_array);

            Matrix<float> hidden = input.dot(weights_input_hidden);
            hidden += bias_hidden;
            hidden.map(sigmoid);

            Matrix<float> output = hidden.dot(weights_hidden_output);
            output += bias_output;
            output.map(sigmoid);

            return { hidden, output };
        }

        TrainMetrics train_step(const std::vector<float>& input_array, const std::vector<float>& target_array) {
            Matrix<float> input(input_array);
            NetworkResults prediction_results = predict(input_array);   
            return { 0, input };
        }
};

int main() {
    std::vector<float> inputs = {1, 0};
    NeuralNetwork brain(2, 2, 1, 0.1);
    NetworkResults results = brain.predict(inputs);
    std::cout << "Network Results: " << std::endl;
    results.output.print();
    return 0; 
};
