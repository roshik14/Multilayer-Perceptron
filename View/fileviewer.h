#ifndef SRC_IMAGEREADER_H_
#define SRC_IMAGEREADER_H_

#include <QFileDialog>
#include <QString>

namespace s21 {
class FileViewer;
}

class s21::FileViewer final {
 public:
  bool ReadImage();
  bool ReadWeights();
  bool ReadDataset();
  bool SaveWeights();
  QString FileName() const;

 private:
  QString ProcessFile(const std::string& caption, const std::string& format,
                      QFileDialog::AcceptMode mode);

 private:
  QString file_name_;
};

#endif  // SRC_IMAGEREADER_H_
