#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /*
  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;
  n_x_ = x_.size();
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2 * n_aug_ + 1);

  x_aug = VectorXd(n_aug_);
  x_sig_aug = MatrixXd(n_aug_ , weights_.size());
  P_aug = MatrixXd(n_aug_ , n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	if(!is_initialized_) Initialize(meas_package);

	else {
		Prediction((meas_package.timestamp_ - time_us_) / 1000000.);
		(meas_package.sensor_type_ == MeasurementPackage::RADAR) ? UpdateRadar(meas_package) : UpdateLidar(meas_package);
		time_us_ = meas_package.timestamp_;
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /*
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

	// CREATE AUGMENTED SIGMA POINTS
	
	// probably faster to zero these than recreate them each time
	x_aug.fill(0);
	x_sig_aug.fill(0);
	P_aug.fill(0);

	// augmented mean state
	x_aug.head(5) = x_;

	// augmented covariance matrix
	P_aug.topLeftCorner(5,5) = P_;
	P_aug(5 , 5) = std_a_*std_a_;
	P_aug(6 , 6) = std_yawdd_*std_yawdd_;

	// sqrt matrix
	L = P_aug.llt().matrixL();

	// augmented sigma points
	x_sig_aug.col(0) = x_aug;
	for(int i=0; i< n_aug_; i++)
	{
		x_sig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		x_sig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}

	// predict sigma points
	double delta_t2 = delta_t*delta_t;

	for(int i = 0; i< weights_.size(); i++)
	{
		double px = x_sig_aug(0 , i);
		double py = x_sig_aug(1 , i);
		double v = x_sig_aug(2 , i);
		double yaw = x_sig_aug(3 , i);
		double yawd = x_sig_aug(4 , i);
		double nu_a = x_sig_aug(5 , i);
		double nu_yawdd = x_sig_aug(6 , i);

		double px_p , py_p;

		// protect against div by 0
		if(fabs(yawd) > 0.001)
		{
			px_p = px + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
			py_p = py + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
		} else
		{
			px_p = px + v*delta_t*cos(yaw);
			py_p = py + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		// add noise
		px_p += 0.5*nu_a*delta_t2*cos(yaw);
		py_p += 0.5*nu_a*delta_t2*sin(yaw);
		v_p *= nu_a*delta_t;
		yaw_p += 0.5*nu_yawdd*delta_t2;
		yawd_p += nu_yawdd*delta_t;

		// write predicted sigma
		Xsig_pred_(0 , i) = px_p;
		Xsig_pred_(1 , i) = py_p;
		Xsig_pred_(2 , i) = v_p;
		Xsig_pred_(3 , i) = yaw_p;
		Xsig_pred_(4 , i) = yawd_p;
	}

	// predicted state mean
	x_ = Xsig_pred_ * weights_;

	// predicted state covariance matrix
	P_.fill(0.);
	for(int i=0; i < weights_.size(); i++)
	{
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		// normalize angles
		while(x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
		while(x_diff(3) <-M_PI) x_diff(3) += 2.*M_PI;

		P_ += weights_(i) * x_diff * x_diff.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  
	VectorXd z = meas_package.raw_measurements_;

	// measurement dimension for lidar
	int n_z = 2;

	// put sigma points into measurement space
	// there must be a better way to write this...
	MatrixXd Zsig = MatrixXd(n_z , weights_.size());
	for(int i=0; i < weights_.size(); i++)
	{
		Zsig(0,i) = Xsig_pred_(0 , i);
		Zsig(1,i) = Xsig_pred_(1 , i);
	}

	VectorXd z_pred = weights_ * Zsig;

	// covariance matrix
	MatrixXd S = MatrixXd(n_z , n_z);
	for(int i=0; i < weights_.size(); i++)
	{
		VectorXd z_diff = Zsig.col(i) - z_pred;
		S += weights_(i) * z_diff * z_diff.transpose();
	}

	// measurement noise
	MatrixXd R = MatrixXd(n_z , n_z);
	R << std_laspx_*std_laspx_ , 0 ,
		0 , std_laspy_*std_laspy_;

	S += R;

	// cross-correlation
	MatrixXd Tc = MatrixXd(n_z , n_z);
	for(int i = 0; i < weights_.size(); i++)
	{
		VectorXd z_diff = Zsig.col(i) - z_pred;
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		Tc += weights_(i) * x_diff * z_diff.transpose();
	}

	// Kalman gain
	MatrixXd K = Tc * S.inverse();

	// residual
	VectorXd z_diff = z - z_pred;

	// update state and covariance
	x_ += K * z_diff;
	P_ -= K * S * K.transpose();

	// calculate NIS

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /*
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

	int n_z = 3;
	VectorXd z = meas_package.raw_measurements_;

	// transform sigam points to measurement space

	MatrixXd Zsig = MatrixXd(n_z, weights_.size());
	for(int i=0; i<weights_.size(); i++){

		double px = Xsig_pred_(0,i);
		double py = Xsig_pred_(1,i);
		double v = Xsig_pred_(2,i);
		double yaw = Xsig_pred_(3,i);

		double  v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		// measurement model
		Zsig(0,i) = sqrt(px*px + py*py);
		Zsig(1,i) = atan2(py, px);
		Zsig(2,i) = (px*v1 + py*v) / Zsig(0,i);
	}
  
	// mean predicted measurement
  	VectorXd z_pred = weights_ * Zsig;
	
	// measurement covariance
	MatrixXd S = MatrixXd(n_z,n_z);
	S.fill(0.0);
	for(int i=0; i<weights_.size(); i++){
		VectorXd z_diff = Zsig.col(i) - z_pred;
		while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
		while(z_diff(1) <-M_PI) z_diff(1) += 2.*M_PI;
		S += weights_(i) * z_diff * z_diff.transpose();
	}
	
	// measurement noise
	MatrixXd R = MatrixXd(n_z,n_z);
	R << std_radr_*std_radr_,	0,	0,
			0,	std_radphi_*std_radphi_,	0,
			0,	0,	std_radrd_*std_radrd_;
  
	S += R;
	
	
	// Update Radar
	
	// cross correlation
	MatrixXd Tc = MatrixXd(n_z,n_z);
	Tc.fill(0.0);
	for(int i = 0; i < weights_.size(); i++)
	{
		VectorXd z_diff = Zsig.col(i) - z_pred;
		while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
		while(z_diff(1) <-M_PI) z_diff(1) += 2.*M_PI;
		

		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		while(x_diff(1) > M_PI) x_diff(1) -= 2.*M_PI;
		while(x_diff(1) <-M_PI) x_diff(1) += 2.*M_PI;
	
		Tc += weights_(i) * x_diff * z_diff.transpose();
	}
	
	// kalman gain
	MatrixXd K = Tc * S.inverse();
	
	// residual
	VectorXd z_diff = z - z_pred;
	
	// normalize angles again
	while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
	while(z_diff(1) <-M_PI) z_diff(1) += 2.*M_PI;
		
	// update state and covariance
	x_ += K * z_diff;
	P_ -= K * S * K.transpose();
	
	
}

void UKF::Initialize(MeasurementPackage& meas_package)
{
	// time
	time_us_ = meas_package.timestamp_;

	// ensure zero state
	x_.fill(0.);

	// state covariance matrix
	P_ << 1 , 0 , 0 , 0 , 0 ,
		0 , 1 , 0 , 0 , 0 ,
		0 , 0 , 1 , 0 , 0 ,
		0 , 0 , 0 , 1 , 0 ,
		0 , 0 , 0 , 0 , 1;

	// weights
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for(int i = 0; i < weights_.size(); i++)
	{
		weights_(i) = 0.5 / (n_aug_ + lambda_);
	}

	// initialize state based on measurement type
	if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
	{
		double rho = meas_package.raw_measurements_(0);
		double theta = meas_package.raw_measurements_(1);

		x_(0) = rho*cos(theta);
		x_(1) = rho*sin(theta);
	}
	else if(meas_package.sensor_type_ == MeasurementPackage::LASER)
	{
		x_(0) = meas_package.raw_measurements_(0);
		x_(1) = meas_package.raw_measurements_(1);

		// protect against 0 positions in initial state
		if(fabs(x_(0) < 0.001) && fabs(x_(1) < 0.001))
		{
			x_(0) = 0.001;
			x_(1) = 0.001;
		}
	}

	// done
	is_initialized_ = true;
}
