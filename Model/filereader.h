#ifndef SRC_MODEL_DATASETREADER_H_
#define SRC_MODEL_DATASETREADER_H_

#include <fstream>
#include <string>
#include <vector>

#include "weights.h"

namespace s21 {
class FileReader;
class DatasetReader;
class WeightsReader;
class WeightsSaver;
}  // namespace s21

class s21::FileReader {
 public:
  virtual bool IsOpen() const = 0;
  virtual ~FileReader() {}
};

class s21::DatasetReader final : public FileReader {
  using IntVector = std::vector<int>;
  using Dataset = std::vector<std::pair<IntVector, int>>;

 public:
  explicit DatasetReader(const std::string &file_name);
  bool IsOpen() const override;
  void Read();
  Dataset data() const;

 private:
  IntVector GetNextLine();
  IntVector ToIntVector(const std::vector<std::string> &data);
  std::vector<std::string> Split(const std::string &str);

 private:
  Dataset data_;
  std::ifstream input_stream_;
};

class s21::WeightsReader final : public FileReader {
 public:
  explicit WeightsReader(const std::string &file_name);
  bool IsOpen() const override;
  Weights ReadWeights();
  bool IsLoadedWeightsCorrect();

 private:
  Weights::RegularWeights ReadInputWeights();
  Weights::HiddenWeights ReadHiddenWeights();
  Weights::RegularWeights ReadOutputWeights();
  Weights::RegularWeights ReadRegularWeights();
  Weights::RegularWeights InitRegularWeights(int rows, int cols);
  Weights::HiddenWeights InitHiddenWeights(int outer_vector_size,
                                           int nested_vector_size,
                                           int second_vector_size);

 private:
  std::ifstream input_stream_;
};

class s21::WeightsSaver final : public FileReader {
 public:
  explicit WeightsSaver(const std::string &file_name);
  bool IsOpen() const override;
  void SaveWeights(const Weights &weights);

 private:
  void SaveRegularWeights(const Weights::RegularWeights &weights);
  void SaveHiddenWeights(const Weights::HiddenWeights &weights);
  void WriteEndLine();

 private:
  std::ofstream output_stream_;
};

#endif  // SRC_MODEL_DATASETREADER_H_
