#include <iomanip>

#include "filereader.h"

using s21::WeightsSaver;

namespace {
std::ofstream &operator<<(std::ofstream &out,
                          const std::pair<double, double> &pair);
}

WeightsSaver::WeightsSaver(const std::string &file_name)
    : output_stream_(file_name) {}

bool WeightsSaver::IsOpen() const { return output_stream_.is_open(); }

void WeightsSaver::SaveWeights(const Weights &weights) {
  SaveRegularWeights(weights.GetInputWeights());
  SaveHiddenWeights(weights.GetHiddenWeights());
  SaveRegularWeights(weights.GetOutputWeights());
}

void WeightsSaver::SaveHiddenWeights(const Weights::HiddenWeights &weights) {
  if (!weights.empty()) {
    auto outer_vector_size = weights.size();
    auto nested_vector_size = weights[0].size();
    auto second_nested_size = weights[0][0].size();
    output_stream_ << outer_vector_size << ' ' << nested_vector_size << ' '
                   << second_nested_size << '\n';
    for (size_t i = 0; i != outer_vector_size; ++i) {
      for (size_t j = 0; j != nested_vector_size; ++j) {
        for (size_t k = 0; k != second_nested_size; ++k) {
          output_stream_ << weights[i][j][k] << ' ';
        }
      }
      WriteEndLine();
    }
    WriteEndLine();
  }
}

void WeightsSaver::SaveRegularWeights(const Weights::RegularWeights &weights) {
  if (!weights.empty()) {
    auto rows = weights.size();
    auto cols = weights[0].size();
    output_stream_ << rows << ' ' << cols << '\n';
    for (size_t i = 0; i != rows; ++i) {
      for (size_t j = 0; j != cols; ++j) {
        output_stream_ << weights[i][j] << ' ';
      }
      WriteEndLine();
    }
  }
}

inline void WeightsSaver::WriteEndLine() { output_stream_ << '\n'; }

namespace {
std::ofstream &operator<<(std::ofstream &out,
                          const std::pair<double, double> &pair) {
  out << std::fixed << std::setprecision(8) << pair.first;
  out << ' ';
  out << std::fixed << std::setprecision(8) << pair.second;
  return out;
}
}  // namespace
