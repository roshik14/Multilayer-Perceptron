#ifndef SRC_TESTDATA_H_
#define SRC_TESTDATA_H_

#include <string>

namespace s21 {
enum class PerceptronType;
struct TrainData;
struct NetworkTestData;
}  // namespace s21

enum class s21::PerceptronType {
  kMatrix,
  kGraph,
};

struct s21::TrainData {
  std::string file_name;
  int epoch_count;
  int neuron_count;
  int layers_count;
  PerceptronType type;
};

struct s21::NetworkTestData {
  double average_accuracy;
  double precision;
  double recall;
  double f_measure;
  double total_elapsed_time;
};

#endif  // SRC_TESTDATA_H_
