#ifndef SRC_MODEL_WEIGHTS_H_
#define SRC_MODEL_WEIGHTS_H_

#include <vector>

namespace s21 {
class Weights;
}

class s21::Weights {
  using WeightsVector = std::vector<std::pair<double, double>>;

 public:
  using RegularWeights = std::vector<WeightsVector>;
  using HiddenWeights = std::vector<RegularWeights>;

 public:
  Weights(int neurons, int layers);
  Weights() = default;
  bool IsEmpty() const noexcept;
  bool IsValid() const noexcept;
  void SetInputLayerWeights(RegularWeights &&weights);
  void SetHiddenLayerWeights(HiddenWeights &&weights);
  void SetOutputLayerWeights(RegularWeights &&weights);
  RegularWeights GetInputWeights() const;
  HiddenWeights GetHiddenWeights() const;
  RegularWeights GetOutputWeights() const;
  WeightsVector::reference InputWeight(int i, int j);
  WeightsVector::const_reference InputWeight(int i, int j) const;
  WeightsVector::reference HiddenWeight(int i, int j, int k);
  WeightsVector::const_reference HiddenWeight(int i, int j, int k) const;
  WeightsVector::reference OutputWeight(int i, int j);
  WeightsVector::const_reference OutputWeight(int i, int j) const;

 public:
  static const int kOutputSize = 26;
  static const int kInputSize = 784;

 private:
  void InitiateInputWeights(int neurons);
  void InitiateHiddenWeights(int neurons, int layers);
  void InitiateOutputWeights(int neurons);

 private:
  RegularWeights input_weights_;
  HiddenWeights hidden_weights_;
  RegularWeights output_weights_;
};

#endif  // SRC_MODEL_WEIGHTS_H_
