// L1_residual_min.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include "L1_residual_min.h"

#define _SCL_SECURE_NO_WARNINGS

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


#include <iostream>	// for debugging


/**
*
* Solves L1 residual minimization problem by Iteratively Reweighted Least-squares (IRLS)
* 
*	Minimize_x ||Ax - b||_1 
*	There is a conversion ||Ax-b||_1 = (Ax-b)^T W^2 (Ax-b) = ||W(Ax-b)||_2^2 with a proper W.
*	IRLS seeks for a good W in an iterative manner and derives an L1 solution.
*
*	Input:	Matrix A \in R^{m x n}
*			Vector b \in R^{m}
*			(Optional)	MAX_ITER: maximum number of iterations 
						tol: tolerance for the termination criterion
*	Output:	Vector x \in R^{n}
*
**/
double L1_residual_min(const Mat &A, const Vec &b, Vec &x, const int MAX_ITER = 1000, const double tol = 1.0e-8)
{
	const double eps = 1.0e-8;	// A constant small number (used for avoiding zero-division)
	double residual = std::numeric_limits<double>::infinity(); // Set the initial value for the residual to be infinity
	const int m = A.rows();		// # of rows of Matrix A 
	const int n = A.cols();		// # of cols of Matrix A
	if (m != b.size())	// m should agree with the size of Vector b (otherwise, throw an exception)
		throw std::invalid_argument("Exception: Inconsistent dimensionality");

	DMat W(m);			// Prepare an (m x m) diagonal weight matrix
	W.setIdentity();	// Initialize the diagonal matrix as identity

	Vec xold = 1000.0 * Vec::Ones(n);	// Buffer for storing previous solution x
	for (int k = 0; k < MAX_ITER; ++k) // Start iteration of IRLS
	{
		// Create a weighted linear system equation Cx = d (corresponds to WAx = Wb)
		Mat C = W * A;
		Vec d = W * b;
		// Solve Cx = d for x
		x = C.colPivHouseholderQr().solve(d);
		// Check convergence
		if ((x - xold).norm() < tol)
			return residual;
		// Update weight matrix W
		Vec e = b - A * x;
		// Update xold
		xold = x;
		residual = e.lpNorm<1>();
		for (int i = 0; i < m; ++i)
		{
			W.diagonal()[i] = 1.0 / max(sqrt(fabs(e[i])), eps);	// W(i,i) = 1.0 / sqrt|e(i)|	(max with eps is used for avoiding zero-division in the case e(i) = 0.0)
		}
	}
	return 0.0;
}


/**
* [Sparse case]
* Solves L1 residual minimization problem by Iteratively Reweighted Least-squares (IRLS) 
*
*	Minimize_x ||Ax - b||_1
*	There is a conversion ||Ax-b||_1 = (Ax-b)^T W^2 (Ax-b) = ||W(Ax-b)||_2^2 with a proper W.
*	IRLS seeks for a good W in an iterative manner and derives an L1 solution.
*
*	Input:	Matrix A \in R^{m x n}
*			Vector b \in R^{m}
*			(Optional)	MAX_ITER: maximum number of iterations
tol: tolerance for the termination criterion
*	Output:	Vector x \in R^{n}
*
**/
double L1_residual_min_sp(const SpMat &A, const Vec &b, Vec &x, const int MAX_ITER = 1000, const double tol = 1.0e-8)
{
	const double eps = 1.0e-8;	// A constant small number (used for avoiding zero-division)
	double residual = std::numeric_limits<double>::infinity(); // Set the initial value for the residual to be infinity
	const int m = A.rows();		// # of rows of Matrix A 
	const int n = A.cols();		// # of cols of Matrix A
	if (m != b.size())	// m should agree with the size of Vector b (otherwise, throw an exception)
		throw std::invalid_argument("Exception: Inconsistent dimensionality");

	DMat W(m);			// Prepare an (m x m) diagonal weight matrix
	W.setIdentity();	// Initialize the diagonal matrix as identity

	Vec xold = 1000.0 * Vec::Ones(n);	// Buffer for storing previous solution x
	for (int k = 0; k < MAX_ITER; ++k) // Start iteration of IRLS
	{
		// Create a normal equation Cx = d
		SpMat C = W * A;
		Vec d = W * b;
		// Solve WAx = Wb for x
		Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
		solver.compute(C);
		x = solver.solve(d);
		// Check convergence
		if ((x - xold).norm() < tol)
			return residual;
		// Update weight matrix W
		Vec e = b - A * x;
		// Update xold
		xold = x;
		residual = e.lpNorm<1>();
		for (int i = 0; i < m; ++i)
		{
			W.diagonal()[i] = 1.0 / max(sqrt(fabs(e[i])), eps);	// W(i,i) = 1.0 / |e(i)|	(max with eps is used for avoiding zero-division in the case e(i) = 0.0)
		}
	}
	return 0.0;
}


// Dense matrix case
void Example01()
{
	const int m = 100;
	const int n = 10;
	Mat A = Mat::Random(m, n);
	Vec x_gt = Vec::Random(n); // Ground truth solution x_gt
	Vec b = A * x_gt;
	// Corrupt by by flipping signs (outlier)
	for (int i = 0; i < 3; ++i)
		b[i] = -b[i];

	// Derive an L1 solution: minimize_x ||Ax - b||_1
	Vec x_l1;
	double l1_residual = L1_residual_min(A, b, x_l1);
	std::cout << "Difference between L1 solution and the ground truth" << std::endl;
	std::cout << x_l1 - x_gt << std::endl; // Output the difference from the ground truth
	std::cout << "---" << std::endl;
	// Example: Comparison with the L2 solution
	Vec x_l2;
	x_l2 = A.colPivHouseholderQr().solve(b);
	std::cout << "Difference between least-squares solution and the ground truth" << std::endl;
	std::cout << x_l2 - x_gt << std::endl; // Output the difference from the ground truth
}
// Sparse matrix case
void Example02()
{
	const int m = 100;
	const int n = 10;
	Mat A = Mat::Random(m, n);
	SpMat Asp = A.sparseView();
	Vec x_gt = Vec::Random(n); // Ground truth solution x_gt
	Vec b = Asp * x_gt;
	// Corrupt by by flipping signs (outlier)
	for (int i = 0; i < 3; ++i)
		b[i] = -b[i];

	// Derive an L1 solution: minimize_x ||Ax - b||_1
	Vec x_l1;
	double l1_residual = L1_residual_min_sp(Asp, b, x_l1);
	std::cout << "Difference between L1 solution and the ground truth" << std::endl;
	std::cout << x_l1 - x_gt << std::endl; // Output the difference from the ground truth
	std::cout << "---" << std::endl;
	// Example: Comparison with the L2 solution
	Vec x_l2;
	x_l2 = A.colPivHouseholderQr().solve(b);
	std::cout << "Difference between least-squares solution and the ground truth" << std::endl;
	std::cout << x_l2 - x_gt << std::endl; // Output the difference from the ground truth
}


