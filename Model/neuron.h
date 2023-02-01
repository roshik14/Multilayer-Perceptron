#ifndef NEURON_H
#define NEURON_H

#include <map>

class Neuron {
 public:
  void SetOut(double out) { this->out_ = out; }
  void SetNet(double net) { this->net_ = net; }
  void SetError(double error) { this->error_ = error; }
  double GetOut() const { return this->out_; }
  double GetNet() const { return this->net_; }
  double GetError() const { return this->error_; }
  virtual ~Neuron() {}

 protected:
  double net_ = 0.0;
  double out_ = 0.0;
  double error_ = 0.0;
};

class GraphNeuron : public Neuron {
 public:
  double Summator();
  void AddLink(std::pair<Neuron *, std::pair<double, double>> link);
  double GetWeight(Neuron *neuron);
  double GetWeightChange(Neuron *neuron);
  double GetWeight(Neuron *neuron) const;
  double GetWeightChange(Neuron *neuron) const;
  void SetWeight(Neuron *neuron, double weight);
  void SetWeightChange(Neuron *neuron, double weight_change);
  void SubtractFromWeight(Neuron *neuron, double value);
  void AddToWeightChange(Neuron *neuron, double value, double alpha);

 private:
  std::map<Neuron *, std::pair<double, double>> links;
};

#endif  // NEURON_H
