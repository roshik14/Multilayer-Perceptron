#include <sstream>

#include "filereader.h"

using s21::DatasetReader;

DatasetReader::DatasetReader(const std::string& file_name)
    : input_stream_(file_name) {}

bool DatasetReader::IsOpen() const { return input_stream_.is_open(); }

void DatasetReader::Read() {
  Dataset dataset;
  auto row = GetNextLine();
  while (!row.empty()) {
    auto answer = *row.begin();
    row.erase(row.begin());
    dataset.push_back(std::make_pair(row, answer));
    row = GetNextLine();
  }
  data_ = std::move(dataset);
}

DatasetReader::Dataset DatasetReader::data() const { return data_; }

DatasetReader::IntVector DatasetReader::GetNextLine() {
  std::string line;
  std::getline(input_stream_, line);
  return !input_stream_.eof() ? ToIntVector(Split(line)) : std::vector<int>{};
}

DatasetReader::IntVector DatasetReader::ToIntVector(
    const std::vector<std::string>& data) {
  IntVector result;
  for (int i = 0, sz = data.size(); i != sz; ++i) {
    result.push_back(std::stoi(data[i]));
  }
  return result;
}

std::vector<std::string> DatasetReader::Split(const std::string& str) {
  std::istringstream token_stream(str);
  std::string line;
  std::vector<std::string> strings;
  while (std::getline(token_stream, line, ',')) strings.push_back(line);
  return strings;
}
