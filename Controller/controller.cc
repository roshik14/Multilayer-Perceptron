#include "controller.h"

using s21::Controller;
using s21::NetworkTestData;

Controller::Controller() : facade_(std::make_unique<Facade>()) {}

char Controller::GetResult(QImage &image) { return facade_->GetResult(image); }

void Controller::TrainPerceptron(const TrainData &train_data) {
  facade_->TrainPerceptron(train_data);
}

bool Controller::LoadWeights(const std::string &file_name,
                             PerceptronType type) {
  return facade_->LoadWeights(file_name, type);
}

void Controller::SaveWeights(const std::string &file_name) {
  facade_->SaveWeights(file_name);
}

bool Controller::IsNetworkTrained() const {
  return facade_->IsNetworkTrained();
}

NetworkTestData Controller::TestNetwork(const std::string &file_name,
                                        double percent) {
  return facade_->TestNetwork(file_name, percent);
}

void Controller::ChangePerceptronRealization(PerceptronType type) {
  facade_->ChangePerceptronRealization(type);
}

void Controller::DoCrossValidationMethod(const std::string &file_name,
                                         int groups_count) {
  facade_->DoCrossValidationMethod(file_name, groups_count);
}

std::vector<std::vector<double>> Controller::GetErrors() const noexcept {
  return facade_->GetErrors();
}
