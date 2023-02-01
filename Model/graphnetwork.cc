#include "graphnetwork.h"

#include <algorithm>
#include <random>

using s21::GraphNeuralNetwork;
using s21::Weights;

GraphNeuralNetwork::GraphNeuralNetwork(int hidden_layers_count,
                                       int neurons_count)
    : NeuralNetwork(hidden_layers_count, neurons_count),
      input_layer_(Weights::kInputSize),
      output_layer_(Weights::kOutputSize) {
  for (int i = 0; i < hidden_layers_count_; ++i) {
    hidden_layers_.push_back(std::vector<GraphNeuron>(neurons_count_));
  }
  const int rate = 50;
  this->LinkWithRandomValues(rate);
}

GraphNeuralNetwork::GraphNeuralNetwork(Weights &&weights) {
  InitInputLayersFromWeights(weights.GetInputWeights());
  InitHiddenLayersFromWeights(weights.GetHiddenWeights());
  InitOutputLayersFromWeights(weights.GetOutputWeights());
  trained_ = true;
}

void GraphNeuralNetwork::InitInputLayersFromWeights(
    const Weights::RegularWeights &input_weights) {
  auto hidden_neurons_count = input_weights.size();
  input_layer_ = std::vector<Neuron>(Weights::kInputSize);
  hidden_layers_ = std::vector<std::vector<GraphNeuron>>{};
  hidden_layers_.push_back(std::vector<GraphNeuron>(hidden_neurons_count));
  for (size_t i = 0; i < hidden_neurons_count; ++i) {
    for (size_t j = 0; j < Weights::kInputSize; ++j) {
      auto current_weight = input_weights[i][j];
      auto input_neuron_pointer = &this->input_layer_[j];
      this->hidden_layers_[0][i].AddLink(
          std::pair{input_neuron_pointer, current_weight});
    }
  }
}

void GraphNeuralNetwork::InitHiddenLayersFromWeights(
    const Weights::HiddenWeights &hidden_weights) {
  this->hidden_layers_count_ = hidden_weights.size() + 1;
  this->neurons_count_ = hidden_weights[0].size();
  for (auto i = 0; i < this->hidden_layers_count_ - 1; ++i) {
    hidden_layers_.push_back(std::vector<GraphNeuron>(this->neurons_count_));
    for (auto j = 0; j < this->neurons_count_; ++j) {
      for (auto k = 0; k < this->neurons_count_; ++k) {
        auto current_weight = hidden_weights[i][j][k];
        auto hidden_neuron_pointer = &this->hidden_layers_[i][k];
        this->hidden_layers_[i + 1][j].AddLink(
            std::pair{hidden_neuron_pointer, current_weight});
      }
    }
  }
}

void GraphNeuralNetwork::InitOutputLayersFromWeights(
    const Weights::RegularWeights &output_weights) {
  output_layer_ = std::vector<GraphNeuron>(Weights::kOutputSize);
  for (auto i = 0; i < Weights::kOutputSize; ++i) {
    for (auto j = 0; j < this->neurons_count_; ++j) {
      auto current_weight = output_weights[i][j];
      auto hidden_neuron_pointer =
          &this->hidden_layers_[this->hidden_layers_count_ - 1][j];
      this->output_layer_[i].AddLink(
          std::pair{hidden_neuron_pointer, current_weight});
    }
  }
}

void GraphNeuralNetwork::LinkWithRandomValues(const int rate) {
  RandomizeInput(rate);
  RandomizeHidden(rate);
  RandomizeOutput(rate);
}

void GraphNeuralNetwork::RandomizeInput(const int rate) {
  for (int i = 0; i < neurons_count_; ++i) {
    for (int j = 0; j < Weights::kInputSize; ++j) {
      double randomWeight = (rand() % (rate * 2)) / 100.0 - rate / 100.0;
      hidden_layers_[0][i].AddLink(
          {&input_layer_[j], std::make_pair(randomWeight, 0.0)});
    }
  }
}

void GraphNeuralNetwork::RandomizeHidden(const int rate) {
  for (int i = 1; i < hidden_layers_count_; ++i) {
    for (int j = 0; j < neurons_count_; ++j) {
      for (int k = 0; k < neurons_count_; ++k) {
        double randomWeight = (rand() % (rate * 2)) / 100.0 - rate / 100.0;
        hidden_layers_[i][j].AddLink(
            {&hidden_layers_[i - 1][k], std::make_pair(randomWeight, 0.0)});
      }
    }
  }
}

void GraphNeuralNetwork::RandomizeOutput(const int rate) {
  for (int i = 0; i < Weights::kOutputSize; ++i) {
    for (int j = 0; j < neurons_count_; ++j) {
      double randomWeight = (rand() % (rate * 2)) / 100.0 - rate / 100.0;
      output_layer_[i].AddLink({&hidden_layers_[hidden_layers_count_ - 1][j],
                                std::make_pair(randomWeight, 0.0)});
    }
  }
}

double GraphNeuralNetwork::GetSumForHiddenInForwardProp(int i, int j) {
  return hidden_layers_[i][j].Summator();
}

double GraphNeuralNetwork::GetSumForOutputInForwardProp(int i) {
  return output_layer_[i].Summator();
}

double GraphNeuralNetwork::GetSumForOutputInBackProp(int i) {
  auto sum = 0.0;
  for (int j = 0; j < Weights::kOutputSize; ++j) {
    sum += this->output_layer_[j].GetError() *
           this->output_layer_[j].GetWeight(
               &hidden_layers_[hidden_layers_count_ - 1][i]);
  }
  return sum;
}

double GraphNeuralNetwork::GetSumForHiddenInBackProp(int i, int j) {
  auto sum = 0.0;
  for (int k = 0; k < neurons_count_; ++k) {
    sum += hidden_layers_[i + 1][k].GetError() *
           hidden_layers_[i + 1][k].GetWeight(&hidden_layers_[i][j]);
  }
  return sum;
}

char GraphNeuralNetwork::ClassificationSymbol(const std::vector<int> &data) {
  this->ForwardPropagationForInput(input_layer_, data);
  this->ForwardPropagationForHidden(
      this, this->hidden_layers_,
      &GraphNeuralNetwork::GetSumForHiddenInForwardProp, 0);
  this->ForwardPropagationForOutput(
      this, this->output_layer_,
      &GraphNeuralNetwork::GetSumForOutputInForwardProp);
  return this->FindMaxClassification(this->output_layer_);
}

void GraphNeuralNetwork::MethodBackPropagation(
    const int answer, std::vector<double> *epoch_errors) {
  BackPropagationForOutput(output_layer_, answer);
  BackPropagationForFirstHidden(this, hidden_layers_,
                                &GraphNeuralNetwork::GetSumForOutputInBackProp);
  BackPropagationForHidden(this, hidden_layers_,
                           &GraphNeuralNetwork::GetSumForHiddenInBackProp);
  this->CalculateWeightFix();
  *epoch_errors = GetErrorsVectorFromOutputLayer(output_layer_);
}

void GraphNeuralNetwork::CalculateWeightFix() {
  auto alpha = 0.7;
  auto n = 0.1;
  CalculateWeightFixForOutputLayer(alpha, n);
  CalculateWeightFixForHiddenLayer(alpha, n);
  CalculateWeightFixForInputLayer(alpha, n);
}

void GraphNeuralNetwork::CalculateWeightFixForOutputLayer(double alpha,
                                                          double n) {
  for (int i = 0; i < Weights::kOutputSize; ++i) {
    for (int j = 0; j < this->neurons_count_; ++j) {
      auto &current = output_layer_[i];
      auto neuron_to = &this->hidden_layers_[this->hidden_layers_count_ - 1][j];
      current.AddToWeightChange(
          neuron_to, (1 - alpha) * n * neuron_to->GetOut() * current.GetError(),
          alpha);
      current.SubtractFromWeight(neuron_to, current.GetWeightChange(neuron_to));
    }
  }
}

void GraphNeuralNetwork::CalculateWeightFixForHiddenLayer(double alpha,
                                                          double n) {
  for (int i = this->hidden_layers_count_ - 1; i >= 1; --i) {
    for (int j = 0; j < this->neurons_count_; ++j) {
      for (int k = 0; k < this->neurons_count_; ++k) {
        auto &current = hidden_layers_[i][j];
        auto neuron_to = &this->hidden_layers_[i - 1][k];
        current.AddToWeightChange(
            neuron_to,
            (1 - alpha) * n * neuron_to->GetOut() * current.GetError(), alpha);
        current.SubtractFromWeight(neuron_to,
                                   current.GetWeightChange(neuron_to));
      }
    }
  }
}

void GraphNeuralNetwork::CalculateWeightFixForInputLayer(double alpha,
                                                         double n) {
  for (int i = 0; i < this->neurons_count_; ++i) {
    for (int j = 0; j < Weights::kInputSize; ++j) {
      auto &hidden_neuron = hidden_layers_[0][i];
      auto neuron_to = &this->input_layer_[j];
      hidden_neuron.AddToWeightChange(
          neuron_to,
          (1 - alpha) * n * neuron_to->GetOut() * hidden_neuron.GetError(),
          alpha);
      hidden_neuron.SubtractFromWeight(
          neuron_to, hidden_neuron.GetWeightChange(neuron_to));
    }
  }
}

Weights GraphNeuralNetwork::GetWeights() {
  Weights weights(this->neurons_count_, this->hidden_layers_count_);
  FillInputWeights(&weights);
  FillHiddenWeights(&weights);
  FillOutputWeights(&weights);
  return weights;
}

void GraphNeuralNetwork::FillInputWeights(Weights *weights) {
  auto input_weights = weights->GetInputWeights();
  for (int i = 0; i < this->neurons_count_; ++i) {
    for (int j = 0; j < Weights::kInputSize; ++j) {
      input_weights[i][j].first =
          this->hidden_layers_[0][i].GetWeight(&this->input_layer_[j]);
      input_weights[i][j].second =
          this->hidden_layers_[0][i].GetWeightChange(&this->input_layer_[j]);
    }
  }
  weights->SetInputLayerWeights(std::move(input_weights));
}

void GraphNeuralNetwork::FillHiddenWeights(Weights *weights) {
  auto hidden_weights = weights->GetHiddenWeights();
  for (int i = 0, sz = hidden_weights.size(); i != sz; ++i) {
    for (int j = 0; j != this->neurons_count_; ++j) {
      for (int k = 0; k != this->neurons_count_; ++k) {
        hidden_weights[i][j][k].first =
            this->hidden_layers_[i + 1][j].GetWeight(
                &this->hidden_layers_[i][k]);
        hidden_weights[i][j][k].second =
            this->hidden_layers_[i + 1][j].GetWeightChange(
                &this->hidden_layers_[i][k]);
      }
    }
  }
  weights->SetHiddenLayerWeights(std::move(hidden_weights));
}

void GraphNeuralNetwork::FillOutputWeights(Weights *weights) {
  auto output_weights = weights->GetOutputWeights();
  for (auto i = 0; i != Weights::kOutputSize; ++i) {
    for (auto j = 0; j != this->neurons_count_; ++j) {
      output_weights[i][j].first = this->output_layer_[i].GetWeight(
          &this->hidden_layers_[this->hidden_layers_count_ - 1][j]);
      output_weights[i][j].second = this->output_layer_[i].GetWeightChange(
          &this->hidden_layers_[this->hidden_layers_count_ - 1][j]);
    }
  }
  weights->SetOutputLayerWeights(std::move(output_weights));
}
