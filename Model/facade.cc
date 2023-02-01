#include "facade.h"

using s21::Facade;
using s21::NetworkTestData;
using s21::NeuralNetwork;

Facade::Facade() {}

namespace {
std::vector<int> ImageToVector(QImage &image);
}

char Facade::GetResult(QImage &image) {
  auto data = ImageToVector(image);
  return perceptron_ && perceptron_->IsTrained()
             ? perceptron_->ClassificationSymbol(data)
             : 0;
}

void Facade::TrainPerceptron(const TrainData &train_data) {
  auto reader = std::make_unique<DatasetReader>(train_data.file_name);
  if (reader->IsOpen()) {
    reader->Read();
    perceptron_ = this->CreateNetwork(train_data.layers_count,
                                      train_data.neuron_count, train_data.type);
    perceptron_->SetDataset(reader->data());
    perceptron_->TrainNetwork(train_data.epoch_count);
  }
}

bool Facade::LoadWeights(const std::string &file_name, PerceptronType type) {
  auto reader = std::make_unique<WeightsReader>(file_name);
  if (reader->IsOpen()) {
    auto weights = reader->ReadWeights();
    if (!weights.IsEmpty()) {
      perceptron_ = this->CreateNetwork(type, std::move(weights));
      return true;
    }
  }
  return false;
}

void Facade::SaveWeights(const std::string &file_name) {
  auto reader = std::make_unique<WeightsSaver>(file_name);
  if (reader->IsOpen()) {
    reader->SaveWeights(perceptron_->GetWeights());
  }
}

bool Facade::IsNetworkTrained() const {
  return perceptron_ && perceptron_->IsTrained();
}

NetworkTestData Facade::TestNetwork(const std::string &file_name,
                                    double percent) {
  auto reader = std::make_unique<DatasetReader>(file_name);
  if (reader->IsOpen()) {
    reader->Read();
    perceptron_->SetDataset(reader->data());
    return perceptron_->TestNetwork(percent);
  }
  return NetworkTestData{0.0, 0.0, 0.0, 0.0, 0.0};
}

void Facade::ChangePerceptronRealization(PerceptronType type) {
  perceptron_ = CreateNetwork(type, perceptron_->GetWeights());
}

void Facade::DoCrossValidationMethod(const std::string &file_name,
                                     int groups_count) {
  auto reader = std::make_unique<DatasetReader>(file_name);
  if (reader->IsOpen()) {
    reader->Read();
    perceptron_->SetDataset(reader->data());
    perceptron_->CrossValidation(groups_count);
  }
}

std::vector<std::vector<double>> Facade::GetErrors() const noexcept {
  return perceptron_->GetErrors();
}

std::unique_ptr<NeuralNetwork> Facade::CreateNetwork(
    int layers, int neurons, PerceptronType type) const {
  if (type == PerceptronType::kGraph)
    return std::make_unique<GraphNeuralNetwork>(layers, neurons);
  return std::make_unique<MatrixNeuralNetwork>(layers, neurons);
}

std::unique_ptr<NeuralNetwork> Facade::CreateNetwork(PerceptronType type,
                                                     Weights &&weights) const {
  if (type == PerceptronType::kGraph)
    return std::make_unique<GraphNeuralNetwork>(std::move(weights));
  return std::make_unique<MatrixNeuralNetwork>(std::move(weights));
}

namespace {
std::vector<int> ImageToVector(QImage &image) {
  image.invertPixels(QImage::InvertRgb);
  auto width = image.width();
  auto height = image.height();
  std::vector<int> result(width * height);
  size_t result_index = 0;
  for (auto i = 0; i != width; ++i) {
    for (auto j = 0; j != height; ++j) {
      result[result_index++] = image.pixelColor(i, j).red();
    }
  }
  return result;
}
}  // namespace
