#ifndef SRC_MODEL_FACADE_H_
#define SRC_MODEL_FACADE_H_

#include <QImage>
#include <memory>
#include <string>

#include "../useful_data.h"
#include "filereader.h"
#include "graphnetwork.h"
#include "matrixnetwork.h"
#include "neuralnetwork.h"
#include "weights.h"

namespace s21 {
class Facade;
}

class s21::Facade final {
 public:
  Facade();
  char GetResult(QImage &image);
  void TrainPerceptron(const TrainData &train_data);
  void SaveWeights(const std::string &file_name);
  bool LoadWeights(const std::string &file_name, PerceptronType type);
  bool IsNetworkTrained() const;
  NetworkTestData TestNetwork(const std::string &file_name, double percent);
  void ChangePerceptronRealization(PerceptronType type);
  void DoCrossValidationMethod(const std::string &file_name, int groups_count);
  std::vector<std::vector<double>> GetErrors() const noexcept;

 private:
  std::unique_ptr<NeuralNetwork> CreateNetwork(int layers, int neurons,
                                               PerceptronType type) const;
  std::unique_ptr<NeuralNetwork> CreateNetwork(PerceptronType type,
                                               Weights &&weights) const;

 private:
  std::unique_ptr<NeuralNetwork> perceptron_;
};

#endif  // SRC_MODEL_FACADE_H_
