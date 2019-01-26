#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /*
    * predict the state
  */

	double vy = x_(3);

	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /*
    * update the state by using Kalman Filter equations
  */

	VectorXd y = z - H_ * x_;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd K = P_ * Ht * S.inverse();

	// new state
	x_ = x_ + (K * y);
	MatrixXd I = MatrixXd::Identity(x_.size() , x_.size());
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /*
    * update the state by using Extended Kalman Filter equations
  */

	double px = x_(0);
	double py = x_(1);
	double vx = x_(2);
	double vy = x_(3);

	double rho = sqrt(px*px + py*py);
	rho = fmax(rho , 0.0001);

	double theta = atan2(py , px);
	double rho_dot = (px*vx + py*vy)/rho;

	VectorXd z_pred(3);
	z_pred << rho , theta , rho_dot;
	
	VectorXd y = z - z_pred;

	if(y[1] > M_PI) y[1] = y[1] - 2.*M_PI;
	else if(y[1] < -M_PI) y[1] = y[1] + 2.*M_PI;

	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd K = P_ * Ht * S.inverse();

	// new state
	x_ = x_ + (K * y);
	MatrixXd I = MatrixXd::Identity(x_.size() , x_.size());
	P_ = (I - K * H_) * P_;
}


