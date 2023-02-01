#include "matrixnetwork.h"

#include <algorithm>
#include <chrono>
#include <random>

using s21::MatrixNeuralNetwork;
using s21::Weights;
MatrixNeuralNetwork::MatrixNeuralNetwork(int hidden_layers_count,
                                         int neurons_count)
    : NeuralNetwork(hidden_layers_count, neurons_count),
      weights_(neurons_count, hidden_layers_count),
      input_layer_(Weights::kInputSize),
      output_layer_(Weights::kOutputSize) {
  InitHiddenLayers();
  this->LinkWithRandomValues(50);
}

MatrixNeuralNetwork::MatrixNeuralNetwork(Weights &&weights)
    : weights_(std::move(weights)),
      input_layer_(Weights::kInputSize),
      output_layer_(Weights::kOutputSize) {
  auto hidden = weights_.GetHiddenWeights();
  this->hidden_layers_count_ = hidden.size() + 1;
  this->neurons_count_ = hidden[0].size();
  InitHiddenLayers();
  trained_ = true;
}

void MatrixNeuralNetwork::InitHiddenLayers() {
  for (int i = 0; i < this->hidden_layers_count_; ++i) {
    this->hidden_layers_.push_back(std::vector<Neuron>(this->neurons_count_));
  }
}

void MatrixNeuralNetwork::LinkWithRandomValues(const int rate) {
  RandomizeInput(rate);
  RandomizeHidden(rate);
  RandomizeOutput(rate);
}

void MatrixNeuralNetwork::RandomizeInput(const int rate) {
  auto input_weights = weights_.GetInputWeights();
  for (int i = 0; i < this->neurons_count_; ++i) {
    for (int j = 0, sz = input_weights[i].size(); j < sz; ++j) {
      input_weights[i][j].first = (rand() % (rate * 2)) / 100.0 - rate / 100.0;
      input_weights[i][j].second = 0.0;
    }
  }
  weights_.SetInputLayerWeights(std::move(input_weights));
}

void MatrixNeuralNetwork::RandomizeHidden(const int rate) {
  auto hidden_weights = weights_.GetHiddenWeights();
  for (int i = 0; i < this->hidden_layers_count_ - 1; ++i) {
    for (int j = 0; j < this->neurons_count_; ++j) {
      for (int k = 0; k < this->neurons_count_; ++k) {
        hidden_weights[i][j][k].first =
            (rand() % (rate * 2)) / 100.0 - rate / 100.0;
        hidden_weights[i][j][k].second = 0.0;
      }
    }
  }
  weights_.SetHiddenLayerWeights(std::move(hidden_weights));
}

void MatrixNeuralNetwork::RandomizeOutput(const int rate) {
  auto output_weights = weights_.GetOutputWeights();
  for (int i = 0, sz = output_weights.size(); i < sz; ++i) {
    for (int j = 0; j < this->neurons_count_; ++j) {
      output_weights[i][j].first = (rand() % (rate * 2)) / 100.0 - rate / 100.0;
      output_weights[i][j].second = 0.0;
    }
  }
  weights_.SetOutputLayerWeights(std::move(output_weights));
}

char MatrixNeuralNetwork::ClassificationSymbol(const std::vector<int> &data) {
  this->ForwardPropagationForInput(input_layer_, data);

  for (int i = 0; i < this->neurons_count_; ++i) {
    auto sum = 0.0;
    for (int j = 0; j < Weights::kInputSize; ++j) {
      sum += this->input_layer_[j].GetOut() * weights_.InputWeight(i, j).first;
    }
    hidden_layers_[0][i].SetNet(sum);
    hidden_layers_[0][i].SetOut(
        this->SigmoidActivation(hidden_layers_[0][i].GetNet()));
  }

  this->ForwardPropagationForHidden(
      this, this->hidden_layers_,
      &MatrixNeuralNetwork::GetSumForHiddenInForwardProp, 1);
  this->ForwardPropagationForOutput(
      this, this->output_layer_,
      &MatrixNeuralNetwork::GetSumForOutputInForwardProp);
  return this->FindMaxClassification(this->output_layer_);
}

double MatrixNeuralNetwork::GetSumForOutputInForwardProp(int i) {
  auto sum = 0.0;
  for (int j = 0; j < this->neurons_count_; ++j) {
    sum += this->hidden_layers_[this->hidden_layers_count_ - 1][j].GetOut() *
           weights_.OutputWeight(i, j).first;
  }
  return sum;
}

double MatrixNeuralNetwork::GetSumForHiddenInForwardProp(int i, int j) {
  auto sum = 0.0;
  for (int k = 0; k < this->neurons_count_; ++k) {
    sum += this->hidden_layers_[i - 1][k].GetOut() *
           weights_.HiddenWeight(i - 1, j, k).first;
  }
  return sum;
}

double MatrixNeuralNetwork::GetSumForHiddenInBackProp(int i, int j) {
  auto sum = 0.0;
  for (int k = 0; k < this->neurons_count_; ++k) {
    sum += this->hidden_layers_[i + 1][k].GetError() *
           weights_.HiddenWeight(i, k, j).first;
  }
  return sum;
}

double MatrixNeuralNetwork::GetSumForOutputInBackProp(int i) {
  auto sum = 0.0;
  for (int j = 0; j < Weights::kOutputSize; ++j) {
    sum +=
        this->output_layer_[j].GetError() * weights_.OutputWeight(j, i).first;
  }
  return sum;
}

void MatrixNeuralNetwork::CalculateWeightFix() {
  auto alpha = 0.7;
  auto n = 0.1;
  CalculateWeightFixForOutputLayer(alpha, n);
  CalculateWeightFixForHiddenLayer(alpha, n);
  CalculateWeightFixForInputLayer(alpha, n);
}

void MatrixNeuralNetwork::CalculateWeightFixForOutputLayer(double alpha,
                                                           double n) {
  for (int i = 0; i < Weights::kOutputSize; ++i) {
    for (int j = 0; j < this->neurons_count_; ++j) {
      auto &current = weights_.OutputWeight(i, j);
      weights_.OutputWeight(i, j).second *= alpha;
      weights_.OutputWeight(i, j).second +=
          (1 - alpha) * n *
          this->hidden_layers_[this->hidden_layers_count_ - 1][j].GetOut() *
          this->output_layer_[i].GetError();
      current.first -= current.second;
    }
  }
}

void MatrixNeuralNetwork::CalculateWeightFixForHiddenLayer(double alpha,
                                                           double n) {
  for (int i = this->hidden_layers_count_ - 2; i >= 0; --i) {
    for (int j = 0; j < this->neurons_count_; ++j) {
      for (int k = 0; k < this->neurons_count_; ++k) {
        auto &current = weights_.HiddenWeight(i, j, k);
        current.second *= alpha;
        current.second += (1 - alpha) * n *
                          this->hidden_layers_[i][k].GetOut() *
                          this->hidden_layers_[i + 1][j].GetError();
        current.first -= current.second;
      }
    }
  }
}

void MatrixNeuralNetwork::CalculateWeightFixForInputLayer(double alpha,
                                                          double n) {
  for (int i = 0; i < this->neurons_count_; ++i) {
    for (int j = 0; j < Weights::kInputSize; ++j) {
      auto &current = weights_.InputWeight(i, j);
      current.second *= alpha;
      current.second += (1 - alpha) * n * this->input_layer_[j].GetOut() *
                        this->hidden_layers_[0][i].GetError();
      current.first -= current.second;
    }
  }
}

void MatrixNeuralNetwork::MethodBackPropagation(
    const int answer, std::vector<double> *epoch_errors) {
  BackPropagationForOutput(output_layer_, answer);
  BackPropagationForFirstHidden(
      this, hidden_layers_, &MatrixNeuralNetwork::GetSumForOutputInBackProp);
  BackPropagationForHidden(this, hidden_layers_,
                           &MatrixNeuralNetwork::GetSumForHiddenInBackProp);
  this->CalculateWeightFix();
  if (epoch_errors)
    *epoch_errors = GetErrorsVectorFromOutputLayer(output_layer_);
}

Weights MatrixNeuralNetwork::GetWeights() { return weights_; }
