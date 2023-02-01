#include "neuron.h"

double GraphNeuron::Summator() {
  double result = 0;
  for (auto link : this->links) {
    result += link.first->GetOut() * link.second.first;
  }
  return result;
}

void GraphNeuron::AddLink(std::pair<Neuron *, std::pair<double, double>> link) {
  this->links.insert(link);
}

double GraphNeuron::GetWeight(Neuron *neuron) {
  return this->links[neuron].first;
}

double GraphNeuron::GetWeightChange(Neuron *neuron) {
  return this->links[neuron].second;
}

double GraphNeuron::GetWeight(Neuron *neuron) const {
  return this->links.find(neuron)->second.first;
}

double GraphNeuron::GetWeightChange(Neuron *neuron) const {
  return this->links.find(neuron)->second.second;
}

void GraphNeuron::SetWeight(Neuron *neuron, double weight) {
  this->links[neuron].first = weight;
}

void GraphNeuron::SetWeightChange(Neuron *neuron, double weight_change) {
  this->links[neuron].second = weight_change;
}

void GraphNeuron::SubtractFromWeight(Neuron *neuron, double value) {
  this->links[neuron].first -= value;
}

void GraphNeuron::AddToWeightChange(Neuron *neuron, double value,
                                    double alpha) {
  this->links[neuron].second *= alpha;
  this->links[neuron].second += value;
}
