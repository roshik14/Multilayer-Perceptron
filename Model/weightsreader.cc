#include "filereader.h"

using s21::Weights;
using s21::WeightsReader;

namespace {
std::ifstream &operator>>(std::ifstream &in, std::pair<double, double> &pair);
}

WeightsReader::WeightsReader(const std::string &file_name)
    : input_stream_(file_name) {}

bool WeightsReader::IsOpen() const { return input_stream_.is_open(); }

Weights WeightsReader::ReadWeights() {
  Weights weights;
  weights.SetInputLayerWeights(ReadInputWeights());
  weights.SetHiddenLayerWeights(ReadHiddenWeights());
  weights.SetOutputLayerWeights(ReadOutputWeights());
  return weights.IsValid() ? weights : Weights{};
}

Weights::RegularWeights WeightsReader::ReadInputWeights() {
  return ReadRegularWeights();
}

Weights::HiddenWeights WeightsReader::ReadHiddenWeights() {
  auto outer_vector_size = 0;
  auto nested_vector_size = 0;
  auto second_nested_size = 0;
  input_stream_ >> outer_vector_size >> nested_vector_size >>
      second_nested_size;
  auto weights = InitHiddenWeights(outer_vector_size, nested_vector_size,
                                   second_nested_size);
  for (auto i = 0; i != outer_vector_size; ++i) {
    for (auto j = 0; j != nested_vector_size; ++j) {
      for (auto k = 0; k != second_nested_size; ++k) {
        input_stream_ >> weights[i][j][k];
      }
    }
  }
  return weights;
}

Weights::HiddenWeights WeightsReader::InitHiddenWeights(
    int outer_vector_size, int nested_vector_size, int second_nested_size) {
  Weights::HiddenWeights weights(outer_vector_size);
  for (auto i = 0; i != outer_vector_size; ++i) {
    weights[i] = InitRegularWeights(nested_vector_size, second_nested_size);
  }
  return weights;
}

Weights::RegularWeights WeightsReader::ReadOutputWeights() {
  return ReadRegularWeights();
}

Weights::RegularWeights WeightsReader::ReadRegularWeights() {
  auto rows = 0;
  auto cols = 0;
  input_stream_ >> rows >> cols;
  auto weights = InitRegularWeights(rows, cols);
  for (auto i = 0; i != rows; ++i) {
    for (auto j = 0; j != cols; ++j) {
      input_stream_ >> weights[i][j];
    }
  }
  return weights;
}

Weights::RegularWeights WeightsReader::InitRegularWeights(int rows, int cols) {
  Weights::RegularWeights weights(rows);
  for (auto i = 0; i != rows; ++i) {
    weights[i] = std::vector<std::pair<double, double>>(cols);
  }
  return weights;
}

namespace {
std::ifstream &operator>>(std::ifstream &in, std::pair<double, double> &pair) {
  in >> pair.first >> pair.second;
  return in;
}
}  // namespace
