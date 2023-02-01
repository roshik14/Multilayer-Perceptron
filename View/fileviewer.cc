#include "fileviewer.h"

#include <QDialog>

using s21::FileViewer;

bool FileViewer::ReadImage() {
  file_name_ =
      ProcessFile("Open image", "Images(*.bmp)", QFileDialog::AcceptOpen);
  return !file_name_.isEmpty();
}

bool FileViewer::ReadWeights() {
  file_name_ =
      ProcessFile("Open file", "Text files(*.txt)", QFileDialog::AcceptOpen);
  return !file_name_.isEmpty();
}

bool FileViewer::ReadDataset() {
  file_name_ =
      ProcessFile("Open file", "Text files(*.csv)", QFileDialog::AcceptOpen);
  return !file_name_.isEmpty();
}

bool FileViewer::SaveWeights() {
  file_name_ =
      ProcessFile("Save file", "Text files(*.txt)", QFileDialog::AcceptSave);
  return !file_name_.isEmpty();
}

QString FileViewer::ProcessFile(const std::string& caption,
                                const std::string& format,
                                QFileDialog::AcceptMode mode) {
  auto fileDialog = std::make_unique<QFileDialog>(
      nullptr, QObject::tr(caption.c_str()), "./", QObject::tr(format.c_str()));
  fileDialog->setAcceptMode(mode);
  fileDialog->setFileMode(QFileDialog::FileMode::ExistingFile);
  return fileDialog->exec() == QDialog::Accepted
             ? fileDialog->selectedFiles().first()
             : "";
}

QString FileViewer::FileName() const { return file_name_; }
