#include "weights.h"

using s21::Weights;

Weights::Weights(int neurons, int layers)
    : input_weights_(neurons), output_weights_(kOutputSize) {
  this->InitiateInputWeights(neurons);
  this->InitiateHiddenWeights(neurons, layers);
  this->InitiateOutputWeights(neurons);
}

void Weights::InitiateInputWeights(int neurons) {
  for (auto i = 0; i < neurons; ++i) {
    this->input_weights_[i] =
        std::vector<std::pair<double, double>>(kInputSize);
  }
}

void Weights::InitiateHiddenWeights(int neurons, int layers) {
  for (int i = 0; i < layers - 1; ++i) {
    this->hidden_weights_.push_back(RegularWeights(neurons));
    for (int j = 0; j < neurons; ++j) {
      this->hidden_weights_[i][j] = WeightsVector(neurons);
    }
  }
}

void Weights::InitiateOutputWeights(int neurons) {
  for (int i = 0; i < kOutputSize; ++i) {
    this->output_weights_[i] = WeightsVector(neurons);
  }
}

Weights::RegularWeights Weights::GetInputWeights() const {
  return input_weights_;
}

Weights::HiddenWeights Weights::GetHiddenWeights() const {
  return hidden_weights_;
}

Weights::RegularWeights Weights::GetOutputWeights() const {
  return output_weights_;
}

void Weights::SetInputLayerWeights(RegularWeights &&weights) {
  input_weights_ = std::move(weights);
}

void Weights::SetHiddenLayerWeights(HiddenWeights &&weights) {
  hidden_weights_ = std::move(weights);
}

void Weights::SetOutputLayerWeights(RegularWeights &&weights) {
  output_weights_ = std::move(weights);
}

Weights::WeightsVector::reference Weights::InputWeight(int i, int j) {
  return input_weights_[i][j];
}

Weights::WeightsVector::const_reference Weights::InputWeight(int i,
                                                             int j) const {
  return input_weights_[i][j];
}

Weights::WeightsVector::reference Weights::HiddenWeight(int i, int j, int k) {
  return hidden_weights_[i][j][k];
}

Weights::WeightsVector::const_reference Weights::HiddenWeight(int i, int j,
                                                              int k) const {
  return hidden_weights_[i][j][k];
}
Weights::WeightsVector::reference Weights::OutputWeight(int i, int j) {
  return output_weights_[i][j];
}

Weights::WeightsVector::const_reference Weights::OutputWeight(int i,
                                                              int j) const {
  return output_weights_[i][j];
}

bool Weights::IsValid() const noexcept {
  return !input_weights_.empty() && input_weights_[0].size() == kInputSize &&
         !hidden_weights_.empty() && output_weights_.size() == kOutputSize;
}

bool Weights::IsEmpty() const noexcept {
  return input_weights_.empty() && hidden_weights_.empty() &&
         output_weights_.empty();
}
