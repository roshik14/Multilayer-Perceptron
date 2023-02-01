#include "mainview.h"

#include <QDialog>
#include <QFileDialog>
#include <QSpinBox>
#include <QTimer>

#include "testdialog.h"
#include "traindialog.h"
#include "trainresultsdialog.h"
#include "ui_mainview.h"

using s21::MainView;

MainView::MainView(QWidget *parent)
    : QMainWindow(parent),
      ui_(std::make_unique<Ui::MainView>()),
      controller_(std::make_unique<Controller>()),
      file_viewer_(std::make_unique<FileViewer>()),
      image_reader_(std::make_unique<QImageReader>()),
      error_change_network_(std::make_unique<QMessageBox>()),
      timer_(std::make_unique<QTimer>()) {
  ui_->setupUi(this);
  ConnectSignals();
  ui_->test_results_frame->setVisible(false);
  ui_->result_label->setVisible(false);
  InitErrorMessageBox();
}

MainView::~MainView() {}

void MainView::ConnectSignals() {
  connect(ui_->clear_button, &QPushButton::clicked, this,
          &MainView::ClearBtnClicked);
  connect(ui_->open_image_button, &QPushButton::clicked, this,
          &MainView::OpenImageBtnClicked);
  connect(ui_->drawer, &Drawer::Release, this, &MainView::EvaluateSlot);
  connect(ui_->save_weights_button, &QPushButton::clicked, this,
          &MainView::SaveWeightsBtnClicked);
  connect(ui_->load_weights_button, &QPushButton::clicked, this,
          &MainView::LoadWeightsBtnClicked);
  connect(ui_->train_button, &QPushButton::clicked, this,
          &MainView::TrainBtnClicked);
  connect(ui_->test_button, &QPushButton::clicked, this,
          &MainView::TestBtnClicked);
  connect(ui_->cross_validation_button, &QPushButton::clicked, this,
          &MainView::CrossValidationBtnClicked);
  connect(ui_->graph_type_button, &QAbstractButton::clicked, this,
          &MainView::ChangeNetworkRealizationSlot);
  connect(ui_->matrix_type_button, &QAbstractButton::clicked, this,
          &MainView::ChangeNetworkRealizationSlot);
}

void MainView::InitErrorMessageBox() {
  error_change_network_->setText("Нейросеть не обучена.");
  error_change_network_->setInformativeText("Хотите обучить нейросеть?");
  auto default_btn =
      error_change_network_->addButton("Да", QMessageBox::AcceptRole);
  error_change_network_->addButton("Нет", QMessageBox::RejectRole);
  error_change_network_->setDefaultButton(default_btn);
}

void MainView::ClearBtnClicked() { ui_->drawer->Clear(); }

void MainView::OpenImageBtnClicked() {
  auto selected = file_viewer_->ReadImage();
  if (selected) {
    UpdateStateFromImage();
  }
}

void MainView::SaveWeightsBtnClicked() {
  if (controller_->IsNetworkTrained()) {
    auto selected = file_viewer_->SaveWeights();
    if (selected) {
      controller_->SaveWeights(file_viewer_->FileName().toStdString());
      ShowMessage("Веса сохранены.", QMessageBox::Information);
    }
  } else {
    if (DidUserChooseToTrain()) {
      emit ui_->train_button->clicked();
    }
  }
}

void MainView::LoadWeightsBtnClicked() {
  auto selected = file_viewer_->ReadWeights();
  if (selected) {
    auto load_result = controller_->LoadWeights(
        file_viewer_->FileName().toStdString(), GetUserChoosedType());
    if (load_result) {
      ShowMessage("Веса загружены.", QMessageBox::Information);
    } else {
      ShowMessage("Некорректные веса", QMessageBox::Critical);
    }
  }
}

void MainView::UpdateStateFromImage() {
  image_reader_->setFileName(file_viewer_->FileName());
  auto image = image_reader_->read();
  if (!image.isNull()) {
    ui_->drawer->SetImage(image.scaled(
        ui_->drawer->width(), ui_->drawer->height(), Qt::KeepAspectRatio));
    emit ui_->drawer->Release();
  }
}

void MainView::EvaluateSlot() {
  auto image = ui_->drawer->image().scaled(kScaledImageSize, kScaledImageSize,
                                           Qt::KeepAspectRatio);
  if (!image.isNull() && controller_->IsNetworkTrained()) {
    if (!ui_->result_label->isVisible()) ui_->result_label->setVisible(true);
    auto answer = controller_->GetResult(image);
    ui_->result_label->setText(QString(QChar(answer).toUpper()));
    auto timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainView::TimeoutSlot);
    auto miliseconds_count = 1000;
    timer->start(miliseconds_count);
    ui_->result_label->setStyleSheet("QLabel { color: green; }");
  }
}

void MainView::TimeoutSlot() { ui_->result_label->setStyleSheet(""); }

void MainView::TrainBtnClicked() {
  auto train_dialog = std::make_unique<TrainDialog>();
  if (train_dialog->exec() == QDialog::Accepted) {
    auto data = train_dialog->data();
    controller_->TrainPerceptron(data);
    if (data.type == PerceptronType::kMatrix) {
      ui_->matrix_type_button->setChecked(true);
    } else {
      ui_->graph_type_button->setChecked(true);
    }
    this->current_perceptron_type_ = data.type;
    ShowTrainResultsDialog(data.epoch_count);
  }
}

void MainView::TestBtnClicked() {
  auto test_dialog = std::make_unique<TestDialog>();
  if (controller_->IsNetworkTrained()) {
    if (test_dialog->exec() == QDialog::Accepted) {
      auto dialog_data = test_dialog->data();
      if (!dialog_data.file_name.isEmpty()) {
        auto test_data = controller_->TestNetwork(
            dialog_data.file_name.toStdString(), dialog_data.percent);
        if (!ui_->test_results_frame->isVisible())
          ui_->test_results_frame->setVisible(true);
        ui_->precision_spinbox->setValue(test_data.precision);
        ui_->recall_spinbox->setValue(test_data.recall);
        ui_->avg_acc_spinbox->setValue(test_data.average_accuracy);
        ui_->f_measure_spinbox->setValue(test_data.f_measure);
        ui_->elapsed_time_spinbox->setValue(test_data.total_elapsed_time);
      }
    }
  } else {
    if (DidUserChooseToTrain()) {
      emit ui_->train_button->clicked();
    }
  }
}

void MainView::CrossValidationBtnClicked() {
  if (controller_->IsNetworkTrained()) {
    auto dialog = std::make_unique<CrossValidationDialog>();
    if (dialog->exec() == QDialog::Accepted) {
      auto file_name = dialog->file_name();
      auto groups_count = dialog->groups_count();
      controller_->DoCrossValidationMethod(file_name.toStdString(),
                                           groups_count);
      ShowMessage("Кросс-валидация завершена", QMessageBox::Information);
    }
  } else {
    if (DidUserChooseToTrain()) {
      emit ui_->train_button->clicked();
    }
  }
}

void MainView::ChangeNetworkRealizationSlot() {
  auto user_choosed = GetUserChoosedType();
  if (controller_->IsNetworkTrained() &&
      user_choosed != this->current_perceptron_type_) {
    controller_->ChangePerceptronRealization(GetUserChoosedType());
    ShowMessage("Реализация перцептрона изменена.", QMessageBox::Information);
    this->current_perceptron_type_ = user_choosed;
  }
}

s21::PerceptronType MainView::GetUserChoosedType() {
  return this->ui_->graph_type_button->isChecked() ? PerceptronType::kGraph
                                                   : PerceptronType::kMatrix;
}

bool MainView::DidUserChooseToTrain() {
  return error_change_network_->exec() == QMessageBox::ButtonRole::AcceptRole;
}

void MainView::ShowMessage(const QString &msg, QMessageBox::Icon icon) {
  auto message_box = std::make_unique<QMessageBox>();
  message_box->setIcon(icon);
  message_box->setText(msg);
  message_box->exec();
}

void MainView::ShowTrainResultsDialog(int epoch_count) {
  auto results_dialog = std::make_unique<TrainResultsDialog>();
  results_dialog->SetRows(epoch_count);
  results_dialog->SetErrors(controller_->GetErrors());
  results_dialog->CreateTable();
  results_dialog->exec();
}
