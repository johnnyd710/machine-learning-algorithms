/*
@author: John DiMatteo
creation date: 03-14-2020
desc:
extreme learning machine using Pytorch Tensors

INPUT
    data        - path to csv, first row header, first column has labels
    numhidden   - number of hidden neurons

OUTPUT
    scores      - Tensor of predictions
*/

#include <torch/torch.h>
#include <iostream>

struct XLM {

    torch::Tensor in_weights;
    torch::Tensor out_weights;

    XLM(int input_size, int no_hidden_units) {
        in_weights = torch::rand({input_size,no_hidden_units});
    }

    void train(torch::Tensor X, torch::Tensor y) {
        std::cout << in_weights << std::endl;
        // dot product between X and the random first layer
        X = torch::matmul(X, in_weights);
        // activation function
        X = torch::relu(X);
        // transpose
        torch::Tensor X_T = torch::t(X);
        out_weights = torch::matmul(
            torch::pinverse(
                torch::matmul(X_T, X)
            ),
            torch::matmul(
                X_T,
                y
            )
        );
    };

    torch::Tensor predict(torch::Tensor X) {
        return torch::matmul(
            torch::relu(torch::matmul(X, in_weights)),
            out_weights
        );
    }
};

int main(){

    int row_size = 8;
    int col_size = 8;
    // int output_size = 1;
    int no_hidden_units = 4;

    auto X = torch::rand({row_size, col_size});
    auto y = torch::randint(0, 2, row_size);

    XLM model(col_size, no_hidden_units);
    model.train(X, y);

    torch::Tensor pred = model.predict(X);

    pred = torch::round(pred);

    std::cout << "Accuracy " << (1 - (torch::sum(pred - y) / row_size)) * 100 << "%" << std::endl;

    return 0;

}