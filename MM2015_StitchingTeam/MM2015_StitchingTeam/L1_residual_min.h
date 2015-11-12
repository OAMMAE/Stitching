#pragma once

#include "stdafx.h"
#include <limits>
#include "Eigen/Dense"
#include "Eigen/Sparse"

typedef Eigen::MatrixXd Mat;	// Dense matrix
typedef Eigen::VectorXd Vec;	// Vector
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DMat;	// Diagonal matrix
typedef Eigen::SparseMatrix<double> SpMat;	// Sparse matrix

double L1_residual_min(const Mat &A, const Vec &b, Vec &x, const int MAX_ITER = 1000, const double tol = 1.0e-8);
double L1_residual_min_sp(const SpMat &A, const Vec &b, Vec &x, const int MAX_ITER = 1000, const double tol = 1.0e-8);
void Example01();
void Example02();