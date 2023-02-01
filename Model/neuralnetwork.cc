#include "neuralnetwork.h"

#include <algorithm>
#include <chrono>
#include <random>

using s21::NeuralNetwork;

NeuralNetwork::NeuralNetwork(int hidden_layers_count, int neurons_count)
    : hidden_layers_count_(hidden_layers_count),
      neurons_count_(neurons_count),
      trained_(false) {}

NeuralNetwork::NeuralNetwork()
    : hidden_layers_count_(0), neurons_count_(0), trained_(false) {}

void NeuralNetwork::SetDataset(Dataset &&dataset) noexcept {
  dataset_ = std::move(dataset);
}

bool NeuralNetwork::IsTrained() const noexcept { return trained_; }

double NeuralNetwork::SigmoidActivation(const double input) const {
  return 1 / (1 + exp(-input));
}

double NeuralNetwork::SigmoidError(const double child_error_sum,
                                   const double neuron_out) const noexcept {
  return child_error_sum * neuron_out * (1 - neuron_out);
}

s21::NetworkTestData NeuralNetwork::TestNetwork(double percent) {
  auto start_time = std::chrono::high_resolution_clock::now();

  auto range = dataset_.size() * percent;
  auto correct = this->Test(this->dataset_.begin(),
                            this->dataset_.begin() + dataset_.size() * percent);

  auto end_time = std::chrono::high_resolution_clock::now();

  auto average_accuracy = correct / range;
  auto precision = correct / (correct + (range - correct));
  auto recall = correct / correct;
  auto f_measure = (2 * precision * recall) / (precision + recall);
  auto elapsed_time =
      std::chrono::duration<double>(end_time - start_time).count();

  dataset_ = Dataset{};
  return NetworkTestData{average_accuracy, precision, recall, f_measure,
                         elapsed_time};
}

double NeuralNetwork::Test(const Dataset::iterator start,
                           const Dataset::iterator finish) {
  double correct = 0;
  for (auto it = start; it != finish; ++it) {
    char symbol = ClassificationSymbol(it->first);
    if (symbol == 'a' + it->second - 1) {
      correct += 1.0;
    }
  }
  return correct;
}

void NeuralNetwork::TrainNetwork(int epoch_count) {
  for (int current_epoch = 0; current_epoch < epoch_count; ++current_epoch) {
    std::vector<double> epoch_errors(Weights::kOutputSize);
    this->Train(this->dataset_.begin(), this->dataset_.end(), &epoch_errors);
    auto rng = std::default_random_engine{};
    std::shuffle(this->dataset_.begin(), this->dataset_.end(), rng);
    errors_.push_back(std::move(epoch_errors));
  }
  dataset_ = Dataset{};
  trained_ = true;
}

double NeuralNetwork::Train(const Dataset::iterator start,
                            const Dataset::iterator finish,
                            std::vector<double> *epoch_errors) {
  double correct = 0;
  for (auto it = start; it != finish; ++it) {
    char symbol = ClassificationSymbol(it->first);
    if (symbol == 'a' + it->second - 1) {
      correct += 1.0;
    }
    this->MethodBackPropagation(it->second, epoch_errors);
  }
  return correct;
}

double NeuralNetwork::CrossValidation(int cross_validation_fold) {
  auto splited_dataset = SplitDataset(cross_validation_fold);
  auto correct = 0.0;
  for (int i = 0; i < cross_validation_fold; ++i) {
    for (int j = 0, sz = splited_dataset.size(); j != sz; ++j) {
      if (j != i) {
        auto &current = splited_dataset[j];
        this->Train(current.begin(), current.end(), nullptr);
      }
    }
    correct += this->Test(splited_dataset[i].begin(), splited_dataset[i].end());
  }
  dataset_ = Dataset{};
  return correct / cross_validation_fold;
}

std::vector<s21::NeuralNetwork::Dataset> NeuralNetwork::SplitDataset(
    int cross_validation_fold) {
  std::vector<Dataset> splited_dataset;
  auto rng = std::default_random_engine{};
  std::shuffle(dataset_.begin(), dataset_.end(), rng);
  double split_size = dataset_.size() / cross_validation_fold;

  auto it = dataset_.begin();
  for (auto i = 0; i < cross_validation_fold; ++i) {
    splited_dataset.push_back(Dataset(it, it + split_size));
    it += split_size;
  }
  return splited_dataset;
}

void NeuralNetwork::ForwardPropagationForInput(std::vector<Neuron> &input_layer,
                                               const std::vector<int> &data) {
  for (size_t i = 0, sz = input_layer.size(); i != sz; ++i) {
    input_layer[i].SetNet(data[i] / 255.0);
    input_layer[i].SetOut(input_layer[i].GetNet());
  }
}

std::vector<std::vector<double>> NeuralNetwork::GetErrors() const noexcept {
  return errors_;
}
