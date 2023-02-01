#ifndef SRC_MODEL_MATRIXNETWORK_H_
#define SRC_MODEL_MATRIXNETWORK_H_

#include <vector>

#include "neuralnetwork.h"
#include "neuron.h"
#include "weights.h"

namespace s21 {
class MatrixNeuralNetwork;
}

class s21::MatrixNeuralNetwork final : public NeuralNetwork {
 public:
  MatrixNeuralNetwork(int hidden_layers_count, int neurons_count);
  explicit MatrixNeuralNetwork(Weights &&weights);
  char ClassificationSymbol(const std::vector<int> &data) override;
  Weights GetWeights() override;

 private:
  void LinkWithRandomValues(const int);
  void MethodBackPropagation(const int answer,
                             std::vector<double> *epoch_count) override;
  void CalculateWeightFix();
  void InitHiddenLayers();

  double GetSumForHiddenInForwardProp(int i, int j) override;
  double GetSumForOutputInForwardProp(int i) override;
  double GetSumForOutputInBackProp(int i) override;
  double GetSumForHiddenInBackProp(int i, int j) override;

  void RandomizeInput(const int rate);
  void RandomizeHidden(const int rate);
  void RandomizeOutput(const int rate);

  void CalculateWeightFixForOutputLayer(double alpha, double n);
  void CalculateWeightFixForHiddenLayer(double alpha, double n);
  void CalculateWeightFixForInputLayer(double alpha, double n);

 private:
  Weights weights_;

  std::vector<Neuron> input_layer_;
  std::vector<std::vector<Neuron>> hidden_layers_;
  std::vector<Neuron> output_layer_;
};

#endif  // SRC_MODEL_MATRIXNETWORK_H_
