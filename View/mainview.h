#ifndef MAINVIEW_H
#define MAINVIEW_H

#include <QImageReader>
#include <QMainWindow>
#include <QMessageBox>
#include <QTimer>
#include <memory>

#include "../Controller/controller.h"
#include "../useful_data.h"
#include "crossvalidationdialog.h"
#include "drawer.h"
#include "fileviewer.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainView;
}
namespace s21 {
class MainView;
}
QT_END_NAMESPACE

class s21::MainView : public QMainWindow {
  Q_OBJECT

 public:
  MainView(QWidget* parent = nullptr);
  ~MainView();

 private:
  void ConnectSignals();
  void InitErrorMessageBox();
  void ShowMessage(const QString& msg, QMessageBox::Icon);
  void ShowTrainResultsDialog(int epoch_count);
  bool DidUserChooseToTrain();
  PerceptronType GetUserChoosedType();
  QWidget* GetBoxForTestsResults(const QString& title, int value);

 private slots:
  void ClearBtnClicked();
  void TrainBtnClicked();
  void TestBtnClicked();
  void OpenImageBtnClicked();
  void SaveWeightsBtnClicked();
  void LoadWeightsBtnClicked();
  void EvaluateSlot();
  void UpdateStateFromImage();
  void TimeoutSlot();
  void ChangeNetworkRealizationSlot();
  void CrossValidationBtnClicked();

 private:
  const int kScaledImageSize = 28;
  std::unique_ptr<Ui::MainView> ui_;
  std::unique_ptr<Controller> controller_;
  std::unique_ptr<FileViewer> file_viewer_;
  std::unique_ptr<QImageReader> image_reader_;
  std::unique_ptr<QMessageBox> error_change_network_;
  std::unique_ptr<QTimer> timer_;
  std::unique_ptr<CrossValidationDialog> cross_validation_dialog_;
  PerceptronType current_perceptron_type_;
};
#endif  // MAINVIEW_H
