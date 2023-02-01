#include <iostream>

#include "Model/filereader.h"
#include "Model/graphnetwork.h"
#include "Model/matrixnetwork.h"
#include "Model/neuralnetwork.h"
#include "Model/neuron.h"
#include "Model/weights.h"
#include "gtest/gtest.h"

using namespace s21;

TEST(FileReaderTest, datasetReader) {
  DatasetReader reader("dataset_reader_test.csv");
  if (reader.IsOpen()) {
    reader.Read();
    auto data = reader.data()[0];
    std::vector<int> expected = {7, 16, 24, 25, 255, 34, 35, 36, 33,
                                 0, 0,  0,  0,  0,   23, 34, 35};
    for (size_t i = 0, sz = expected.size(); i != sz; ++i) {
      try {
        ASSERT_EQ(data.first.at(i), expected.at(i));
      } catch (const std::out_of_range &err) {
        FAIL();
      }
    }
    auto answer = 23;
    ASSERT_EQ(data.second, answer);
  } else {
    FAIL();
  }
}

namespace {
void ComparePair(std::pair<double, double> left,
                 std::pair<double, double> right) {
  ASSERT_DOUBLE_EQ(left.first, right.first);
  ASSERT_DOUBLE_EQ(left.second, right.second);
}

void RegularWeightsTest(const Weights::RegularWeights &expected,
                        const Weights::RegularWeights &result) {
  for (int i = 0, sz = expected.size(); i != sz; ++i) {
    for (int j = 0, ssz = expected[i].size(); j != ssz; ++j) {
      try {
        ComparePair(expected.at(i).at(j), result.at(i).at(j));
      } catch (const std::out_of_range &err) {
        FAIL();
      }
    }
  }
}

void HiddenWeightsTest(const Weights::HiddenWeights &expected,
                       const Weights::HiddenWeights &result) {
  for (int i = 0, isz = expected.size(); i != isz; ++i) {
    for (int j = 0, jsz = expected[i].size(); j != jsz; ++j) {
      for (int k = 0, ksz = expected[i][j].size(); k != ksz; ++k) {
        try {
          ComparePair(expected.at(i).at(j).at(k), result.at(i).at(j).at(k));
        } catch (const std::out_of_range &err) {
          FAIL();
        }
      }
    }
  }
}
}  // namespace

TEST(FileReaderTest, incorrectWeights) {
  WeightsReader reader("weights_reader_test.txt");
  if (reader.IsOpen()) {
    auto weights = reader.ReadWeights();
    ASSERT_TRUE(weights.IsEmpty());
  } else {
    FAIL();
  }
}

TEST(MatrixNetworkTest, loadWeights) {
  WeightsReader reader("91percents.txt");
  if (reader.IsOpen()) {
    auto reader_weights = reader.ReadWeights();
    auto tmp = reader_weights;
    MatrixNeuralNetwork network(std::move(tmp));
    auto network_weights = network.GetWeights();
    RegularWeightsTest(reader_weights.GetInputWeights(),
                       network_weights.GetInputWeights());
    HiddenWeightsTest(reader_weights.GetHiddenWeights(),
                      network_weights.GetHiddenWeights());
    RegularWeightsTest(reader_weights.GetOutputWeights(),
                       network_weights.GetOutputWeights());
  } else {
    FAIL();
  }
}

TEST(MatrixNetworkTest, classificationSymbol) {
  WeightsReader reader("91percents.txt");
  if (reader.IsOpen()) {
    MatrixNeuralNetwork network(reader.ReadWeights());
    std::vector<int> test{
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   20,
        27,  8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,   33,  84,  169,
        190, 126, 33,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   47,  190, 233, 251,
        253, 244, 163, 33,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   20,  67,  175, 247, 254, 254,
        254, 254, 244, 127, 10,  1,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   45,  122, 231, 253, 254, 255,
        255, 254, 251, 175, 34,  7,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   4,   114, 203, 254, 254, 254, 253,
        254, 254, 254, 243, 113, 32,  0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   22,  145, 222, 254, 253, 228, 218,
        242, 254, 254, 243, 113, 32,  0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   4,   110, 232, 249, 253, 219, 126, 213,
        246, 254, 252, 177, 34,  7,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   21,  172, 252, 254, 232, 98,  91,  233,
        252, 254, 245, 115, 4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   32,  203, 254, 254, 209, 54,  118, 245,
        254, 254, 232, 82,  2,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   37,  217, 254, 250, 142, 77,  188, 252,
        254, 252, 172, 22,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   37,  214, 232, 207, 111, 177, 250, 254,
        254, 250, 130, 5,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   20,  138, 93,  51,  119, 243, 254, 255,
        255, 252, 173, 22,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   1,   15,  6,   4,   93,  236, 254, 254,
        254, 254, 232, 100, 32,  17,  0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   53,  165, 215, 222,
        233, 251, 253, 232, 152, 75,  2,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   18,  81,  125, 140,
        173, 232, 253, 250, 170, 81,  2,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   4,   9,
        22,  100, 230, 250, 188, 108, 9,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   20,  111, 190, 243, 218, 77,  2,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   7,   115, 242, 243, 112, 3,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   33,  160, 188, 51,  0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   8,   92,  145, 50,  0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   33,  106, 106, 4,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   8,   42,  112, 15,  0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   5,   24,  4,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
    auto answer = 'a';
    ASSERT_EQ(answer, network.ClassificationSymbol(test));
  }
}

TEST(MatrixNetworkTest, networkTesting) {
  WeightsReader reader("91percents.txt");
  if (reader.IsOpen()) {
    DatasetReader dataset_reader(
        "../datasets/emnist-letters/emnist-letters-test.csv");
    if (dataset_reader.IsOpen()) {
      dataset_reader.Read();
      MatrixNeuralNetwork network(reader.ReadWeights());
      network.SetDataset(dataset_reader.data());
      auto data = network.TestNetwork(0.2);
      ASSERT_TRUE(data.average_accuracy > 0.7);
    } else {
      FAIL();
    }
  } else {
    FAIL();
  }
}

TEST(MatrixNeuralNetwork, train) {
  DatasetReader dataset_reader("train_test.csv");
  if (dataset_reader.IsOpen()) {
    dataset_reader.Read();
    MatrixNeuralNetwork network(2, 100);
    network.SetDataset(dataset_reader.data());
    network.TrainNetwork(3);
    DatasetReader test_reader(
        "../datasets/emnist-letters/emnist-letters-test.csv");
    if (test_reader.IsOpen()) {
      test_reader.Read();
      network.SetDataset(test_reader.data());
      auto data = network.TestNetwork(0.1);
      ASSERT_TRUE(data.average_accuracy > 0);
    } else {
      FAIL();
    }
  } else {
    FAIL();
  }
}

TEST(MatrixNetworkTest, train) {}

TEST(GraphNeuralNetwork, loadWeights) {
  WeightsReader reader("91percents.txt");
  if (reader.IsOpen()) {
    auto reader_weights = reader.ReadWeights();
    auto tmp = reader_weights;
    GraphNeuralNetwork network(std::move(tmp));
    auto network_weights = network.GetWeights();
    RegularWeightsTest(reader_weights.GetInputWeights(),
                       network_weights.GetInputWeights());
    HiddenWeightsTest(reader_weights.GetHiddenWeights(),
                      network_weights.GetHiddenWeights());
    RegularWeightsTest(reader_weights.GetOutputWeights(),
                       network_weights.GetOutputWeights());
  } else {
    FAIL();
  }
}

TEST(GraphNeuralNetwork, classificationSymbol) {
  WeightsReader reader("91percents.txt");
  if (reader.IsOpen()) {
    MatrixNeuralNetwork network(reader.ReadWeights());
    std::vector<int> test{
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   4,   21,  20,  1,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   1,   33,  115, 172, 153, 20,  0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   8,   127, 221, 253, 254, 215, 37,  0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   1,   22,  95,  244, 254, 254, 247, 154, 20,  0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   1,   35,  159, 232, 254, 254, 244, 163, 22,  1,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   35,  163, 245, 254, 253, 221, 127, 33,  0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        1,   35,  219, 253, 254, 246, 127, 33,  1,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   7,
        47,  164, 253, 254, 247, 164, 10,  0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   10,  91,
        207, 247, 254, 254, 220, 52,  0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   123, 221,
        253, 254, 254, 252, 172, 21,  0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   20,  52,  231, 254,
        254, 254, 254, 245, 115, 4,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   1,   34,  96,  218, 250, 254, 246,
        196, 237, 254, 217, 39,  0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   3,   36,  202, 234, 253, 254, 244, 163,
        73,  218, 254, 215, 37,  0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   2,   65,  175, 253, 254, 254, 251, 131, 34,
        39,  217, 252, 172, 21,  0,   0,   0,   0,   2,   4,   0,   0,   0,
        0,   0,   0,   0,   0,   5,   126, 248, 254, 254, 246, 206, 23,  1,
        37,  217, 250, 129, 5,   0,   0,   8,   34,  82,  113, 32,  0,   0,
        0,   0,   0,   0,   3,   36,  177, 252, 253, 221, 128, 46,  1,   0,
        37,  217, 250, 129, 14,  32,  51,  127, 204, 233, 241, 113, 4,   0,
        0,   0,   0,   3,   123, 231, 253, 251, 163, 35,  1,   0,   4,   4,
        41,  217, 254, 236, 222, 245, 250, 251, 250, 250, 229, 109, 3,   0,
        0,   0,   3,   67,  222, 254, 252, 191, 55,  38,  37,  51,  125, 127,
        146, 236, 254, 254, 254, 252, 243, 177, 129, 127, 81,  20,  0,   0,
        0,   0,   34,  175, 254, 254, 252, 193, 173, 215, 217, 222, 249, 250,
        250, 254, 251, 233, 216, 172, 113, 34,  5,   4,   2,   0,   0,   0,
        0,   3,   111, 243, 254, 255, 254, 252, 252, 254, 252, 250, 249, 233,
        217, 217, 170, 84,  38,  21,  4,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   33,  158, 222, 240, 222, 217, 172, 127, 82,  39,  37,  21,
        5,   4,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   1,   20,  50,  100, 50,  37,  21,  5,   2,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   3,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
    auto answer = 'a';
    ASSERT_EQ(answer, network.ClassificationSymbol(test));
  }
}

TEST(GraphNeuralNetwork, networkTesting) {
  WeightsReader reader("91percents.txt");
  if (reader.IsOpen()) {
    DatasetReader dataset_reader(
        "../datasets/emnist-letters/emnist-letters-test.csv");
    if (dataset_reader.IsOpen()) {
      dataset_reader.Read();
      GraphNeuralNetwork network(reader.ReadWeights());
      network.SetDataset(dataset_reader.data());
      auto data = network.TestNetwork(0.2);
      ASSERT_TRUE(data.average_accuracy > 0.7);
    } else {
      FAIL();
    }
  } else {
    FAIL();
  }
}

TEST(GraphNeuralNetwork, train) {
  DatasetReader dataset_reader("train_test.csv");
  if (dataset_reader.IsOpen()) {
    dataset_reader.Read();
    GraphNeuralNetwork network(2, 100);
    network.SetDataset(dataset_reader.data());
    network.TrainNetwork(3);
    DatasetReader test_reader(
        "../datasets/emnist-letters/emnist-letters-test.csv");
    if (test_reader.IsOpen()) {
      test_reader.Read();
      network.SetDataset(test_reader.data());
      auto data = network.TestNetwork(0.1);
      ASSERT_TRUE(data.average_accuracy > 0);
    } else {
      FAIL();
    }
  } else {
    FAIL();
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
