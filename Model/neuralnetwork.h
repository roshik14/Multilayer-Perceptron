#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

#include "../useful_data.h"
#include "neuron.h"
#include "weights.h"

namespace s21 {
class NeuralNetwork;
}

class s21::NeuralNetwork {
 protected:
  using Dataset = std::vector<std::pair<std::vector<int>, int>>;

 public:
  virtual void TrainNetwork(int epoch_count);
  virtual double CrossValidation(int cross_validation_fold);
  virtual NetworkTestData TestNetwork(double percent);
  virtual char ClassificationSymbol(const std::vector<int>& data) = 0;
  virtual Weights GetWeights() = 0;
  std::vector<std::vector<double>> GetErrors() const noexcept;
  void SetDataset(Dataset&& dataset) noexcept;
  bool IsTrained() const noexcept;
  virtual ~NeuralNetwork() {}

 protected:
  NeuralNetwork(int hidden_layers_count, int neurons_count);
  NeuralNetwork();
  virtual void MethodBackPropagation(const int answer,
                                     std::vector<double>* epoch_errors) = 0;
  virtual double Train(const Dataset::iterator start,
                       const Dataset::iterator finish,
                       std::vector<double>* epoch_errors);
  virtual double Test(const Dataset::iterator start,
                      const Dataset::iterator finish);
  virtual double GetSumForHiddenInForwardProp(int i, int j) = 0;
  virtual double GetSumForOutputInForwardProp(int i) = 0;
  virtual double GetSumForHiddenInBackProp(int i, int j) = 0;
  virtual double GetSumForOutputInBackProp(int i) = 0;
  double SigmoidActivation(const double input) const;
  double SigmoidError(const double child_error_sum,
                      const double neuron_out) const noexcept;
  void ForwardPropagationForInput(std::vector<Neuron>& input_layer,
                                  const std::vector<int>& data);
  template <typename Neurons>
  void BackPropagationForOutput(Neurons& output_layer, const int answer);
  template <typename Neurons>
  char FindMaxClassification(const Neurons& output_layer);
  template <typename ObjPointer, typename Layer, typename SumFunction>
  void ForwardPropagationForHidden(ObjPointer obj, Layer& hidden_layers,
                                   SumFunction GetSum, int start);
  template <typename ObjPointer, typename Layer, typename SumFunction>
  void ForwardPropagationForOutput(ObjPointer obj, Layer& hidden_layers,
                                   SumFunction GetSum);
  template <typename ObjPointer, typename Layer, typename SumFunction>
  void BackPropagationForFirstHidden(ObjPointer obj, Layer& hidden_layers,
                                     SumFunction GetSum);
  template <typename ObjPointer, typename Layer, typename SumFunction>
  void BackPropagationForHidden(ObjPointer obj, Layer& hidden_layers,
                                SumFunction GetSum);
  template <typename OutputLayer>
  std::vector<double> GetErrorsVectorFromOutputLayer(
      const OutputLayer& source) const;

 private:
  std::vector<Dataset> SplitDataset(int cross_validation_fold);

 protected:
  Dataset dataset_;
  int hidden_layers_count_;
  int neurons_count_;
  bool trained_;
  std::vector<std::vector<double>> errors_;
};

template <typename Neurons>
char s21::NeuralNetwork::FindMaxClassification(const Neurons& output_layer) {
  auto max = output_layer[0].GetOut();
  auto maxIter = 0.0;
  for (int i = 1, sz = output_layer.size(); i < sz; ++i) {
    double val = output_layer[i].GetOut();
    if (val > max) {
      max = val;
      maxIter = i;
    }
  }
  return 'a' + maxIter;
}

template <typename Layer>
void s21::NeuralNetwork::BackPropagationForOutput(Layer& output_layer,
                                                  const int answer) {
  for (int i = 0, sz = output_layer.size(); i < sz; ++i) {
    output_layer[i].SetError(output_layer[i].GetOut() - 0);
  }
  output_layer[answer - 1].SetError(output_layer[answer - 1].GetOut() - 1);
}

template <typename ObjPointer, typename Layer, typename SumFunction>
void s21::NeuralNetwork::ForwardPropagationForHidden(ObjPointer obj,
                                                     Layer& hidden_layers,
                                                     SumFunction GetSum,
                                                     int start) {
  for (int i = start; i != this->hidden_layers_count_; ++i) {
    for (int j = 0; j != this->neurons_count_; ++j) {
      auto sum = (obj->*GetSum)(i, j);
      hidden_layers[i][j].SetNet(sum);
      hidden_layers[i][j].SetOut(
          this->SigmoidActivation(hidden_layers[i][j].GetNet()));
    }
  }
}

template <typename ObjPointer, typename Layer, typename SumFunction>
void s21::NeuralNetwork::ForwardPropagationForOutput(ObjPointer obj,
                                                     Layer& output_layers,
                                                     SumFunction GetSum) {
  for (int i = 0, sz = output_layers.size(); i != sz; ++i) {
    auto sum = (obj->*GetSum)(i);
    auto& current = output_layers[i];
    current.SetNet(sum);
    current.SetOut(this->SigmoidActivation(current.GetNet()));
  }
}

template <typename ObjPointer, typename Layer, typename SumFunction>
void s21::NeuralNetwork::BackPropagationForFirstHidden(ObjPointer obj,
                                                       Layer& hidden_layers,
                                                       SumFunction GetSum) {
  for (int i = 0; i < this->neurons_count_; ++i) {
    auto sum = (obj->*GetSum)(i);
    auto& current = hidden_layers[hidden_layers_count_ - 1][i];
    current.SetError(this->SigmoidError(sum, current.GetOut()));
  }
}

template <typename ObjPointer, typename Layer, typename SumFunction>
void s21::NeuralNetwork::BackPropagationForHidden(ObjPointer obj,
                                                  Layer& hidden_layers,
                                                  SumFunction GetSum) {
  for (int i = this->hidden_layers_count_ - 2; i >= 0; --i) {
    for (int j = 0; j < this->neurons_count_; ++j) {
      auto sum = (obj->*GetSum)(i, j);
      auto& current = hidden_layers[i][j];
      current.SetError(this->SigmoidError(sum, current.GetOut()));
    }
  }
}

template <typename OutputLayer>
std::vector<double> s21::NeuralNetwork::GetErrorsVectorFromOutputLayer(
    const OutputLayer& output_layer) const {
  std::vector<double> errors(output_layer.size());
  for (size_t i = 0, sz = output_layer.size(); i != sz; ++i) {
    errors[i] = output_layer[i].GetError();
  }
  return errors;
}

#endif  // NEURALNETWORK_H
