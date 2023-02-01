#ifndef SRC_MODEL_GRAPHNETWORK_H_
#define SRC_MODEL_GRAPHNETWORK_H_

#include <vector>

#include "neuralnetwork.h"
#include "neuron.h"
#include "weights.h"

namespace s21 {
class GraphNeuralNetwork;
}

class s21::GraphNeuralNetwork final : public NeuralNetwork {
 public:
  GraphNeuralNetwork(int hidden_layers_count, int neurons_count);
  explicit GraphNeuralNetwork(Weights &&weights);
  char ClassificationSymbol(const std::vector<int> &data) override;
  Weights GetWeights() override;

 private:
  void LinkWithRandomValues(const int rate);
  void MethodBackPropagation(const int answer,
                             std::vector<double> *epoch_errors) override;
  void CalculateWeightFix();
  void InitInputLayersFromWeights(const Weights::RegularWeights &weights);
  void InitHiddenLayersFromWeights(const Weights::HiddenWeights &weights);
  void InitOutputLayersFromWeights(const Weights::RegularWeights &weights);

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

  void FillInputWeights(Weights *weights);
  void FillHiddenWeights(Weights *weights);
  void FillOutputWeights(Weights *weights);

 private:
  std::vector<Neuron> input_layer_;
  std::vector<std::vector<GraphNeuron>> hidden_layers_;
  std::vector<GraphNeuron> output_layer_;
};

#endif  // SRC_MODEL_GRAPHNETWORK_H_
