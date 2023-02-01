#include "drawer.h"

#include <QGraphicsEllipseItem>

using s21::Drawer;

Drawer::Drawer(QWidget* parent)
    : QGraphicsView(parent),
      width_(510),
      height_(510),
      scene_(new QGraphicsScene(0, 0, width_, height_)),
      last_pos_(0, 0) {
  setScene(scene_);
}

void Drawer::Clear() {
  scene_->clear();
  scene_->setForegroundBrush(QBrush());
}

void Drawer::SetImage(QImage&& image) noexcept {
  scene_->clear();
  scene_->setForegroundBrush(image);
}

void Drawer::mousePressEvent(QMouseEvent* event) {
  if (IsLeftButton(event)) {
    last_pos_ = event->pos();
    auto circle = new QGraphicsEllipseItem(
        last_pos_.x() - 25, last_pos_.y() - 25, kPointSize, kPointSize);
    circle->setBrush(Qt::black);
    scene_->addItem(circle);
  }
}

void Drawer::mouseMoveEvent(QMouseEvent* event) {
  if (IsLeftButton(event)) {
    auto pos = event->pos();
    scene_->addLine(last_pos_.x(), last_pos_.y(), pos.x(), pos.y(),
                    QPen(Qt::black, kPointSize, Qt::SolidLine, Qt::RoundCap));
    last_pos_ = event->pos();
  }
}

void Drawer::mouseReleaseEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    Q_UNUSED(event);
    emit Release();
  }
}

bool Drawer::IsLeftButton(QMouseEvent* event) const {
  return event->buttons() == Qt::LeftButton;
}

QImage Drawer::image() { return grab().toImage(); }
