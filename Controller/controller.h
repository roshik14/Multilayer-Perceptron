#ifndef SRC_CONTROLLER_CONTROLLER_H_
#define SRC_CONTROLLER_CONTROLLER_H_

#include <QImage>
#include <memory>
#include <string>

#include "../Model/facade.h"
#include "../useful_data.h"

namespace s21 {
class Controller;
}

class s21::Controller final {
 public:
  Controller();
  char GetResult(QImage &image);
  void TrainPerceptron(const TrainData &train_data);
  void SaveWeights(const std::string &file_name);
  bool LoadWeights(const std::string &file_name, PerceptronType type);
  bool IsNetworkTrained() const;
  void ChangePerceptronRealization(PerceptronType type);
  void DoCrossValidationMethod(const std::string &file_name, int groups_count);
  NetworkTestData TestNetwork(const std::string &file_name, double percent);
  std::vector<std::vector<double>> GetErrors() const noexcept;

 private:
  std::unique_ptr<Facade> facade_;
};

#endif  // SRC_CONTROLLER_CONTROLLER_H_
