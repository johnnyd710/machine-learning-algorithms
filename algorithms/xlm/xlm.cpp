#include <torch/torch.h>
#include <iostream>

#define NO_HIDDEN_LAYERS 100

// build a neural network similar to how you would do it with Pytorch 

struct Model : torch::nn::Module {

    // Constructor
    Model() {
        // construct and register your layers
        in = register_module("in",torch::nn::Linear(8,NO_HIDDEN_LAYERS));
        h = register_module("h",torch::nn::Linear(NO_HIDDEN_LAYERS,NO_HIDDEN_LAYERS));
        out = register_module("out",torch::nn::Linear(NO_HIDDEN_LAYERS,1));
    }

    // the forward operation (how data will flow from layer to layer)
    torch::Tensor forward(torch::Tensor X){
        // let's pass relu 
        X = torch::relu(in->forward(X));
        X = torch::relu(h->forward(X));
        X = torch::sigmoid(out->forward(X));
        
        // return the output
        return X;
    }

    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};



};


int main(){

    Model model;
    
    auto in = torch::rand({8,});

    auto out = model.forward(in);

    std::cout << in << std::endl;
    std::cout << out << std::endl;

}