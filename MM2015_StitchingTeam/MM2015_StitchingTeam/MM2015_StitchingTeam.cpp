// MM2015_StitchingTeam.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//


#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <math.h>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Dense>

//.lib ファイルの指定
#ifdef _DEBUG
#pragma comment(lib,"opencv_calib3d2411d.lib")
#pragma comment(lib,"opencv_contrib2411d.lib")
#pragma comment(lib,"opencv_core2411d.lib")
#pragma comment(lib,"opencv_features2d2411d.lib")
#pragma comment(lib,"opencv_flann2411d.lib")
#pragma comment(lib,"opencv_gpu2411d.lib")
#pragma comment(lib,"opencv_highgui2411d.lib")
#pragma comment(lib,"opencv_imgproc2411d.lib")
#pragma comment(lib,"opencv_legacy2411d.lib")
#pragma comment(lib,"opencv_ml2411d.lib")
#pragma comment(lib,"opencv_nonfree2411d.lib")
#pragma comment(lib,"opencv_objdetect2411d.lib")
#pragma comment(lib,"opencv_ocl2411d.lib")
#pragma comment(lib,"opencv_photo2411d.lib")
#pragma comment(lib,"opencv_stitching2411d.lib")
#pragma comment(lib,"opencv_superres2411d.lib")
#pragma comment(lib,"opencv_ts2411d.lib")
#pragma comment(lib,"opencv_video2411d.lib")
#pragma comment(lib,"opencv_videostab2411d.lib")
#else
#pragma comment(lib,"opencv_calib3d2411.lib")
#pragma comment(lib,"opencv_contrib2411.lib")
#pragma comment(lib,"opencv_core2411.lib")
#pragma comment(lib,"opencv_features2d2411.lib")
#pragma comment(lib,"opencv_flann2411.lib")
#pragma comment(lib,"opencv_gpu2411.lib")
#pragma comment(lib,"opencv_highgui2411.lib")
#pragma comment(lib,"opencv_imgproc2411.lib")
#pragma comment(lib,"opencv_legacy2411.lib")
#pragma comment(lib,"opencv_ml2411.lib")
#pragma comment(lib,"opencv_nonfree2411.lib")
#pragma comment(lib,"opencv_objdetect2411.lib")
#pragma comment(lib,"opencv_ocl2411.lib")
#pragma comment(lib,"opencv_photo2411.lib")
#pragma comment(lib,"opencv_stitching2411.lib")
#pragma comment(lib,"opencv_superres2411.lib")
#pragma comment(lib,"opencv_ts2411.lib")
#pragma comment(lib,"opencv_video2411.lib")
#pragma comment(lib,"opencv_videostab2411.lib")
#endif
//.lib ファイルの指定

using namespace cv;
using namespace Eigen;

using cv::imread;
using cv::Mat;
using std::vector;
using cv::imshow;
using cv::waitKey;
using cv::Size;
using cv::Vec3b;
using cv::imwrite;

typedef vector<vector<Point2f>> matKey;
typedef vector<vector<int>> Index;
typedef Eigen::Triplet<double> T;

/*初期設定*/
const int count = 2;
const int dodo = 0;  //0ならransac 1なら普通のやつ 3ならhomoの平均 2ならhomo計算の実装 7はhomographyの計算テスト 6もhomoテストだが500点
std::string HomeDir = "D:\Stitching/datadebug/";
const double p = 1.3; //画像の拡大量

/*関数プロトタイプ宣言*/
int calchomo_x(Mat homo, int x, int y);
int calchomo_y(Mat homo, int x, int y);
Mat homography(Mat homo, Mat BasePic, Mat result, int k);
matKey FindKeyPoint(Mat input1, Mat input2);
Mat marginCut(Mat src);
void mix2Pic(Mat &BasePic, Mat &AddPic, int k);
void zoomPicture(cv::Mat src, cv::Mat dst, cv::Point2i center, double rate);
Mat makePicture(cv::Mat src, double space);
Mat matching(Mat input10, Mat input20, int k);
Mat stitching2Pic(Mat &BasePic1, Mat &AddPic1, int k);
Mat calcHomo(Mat &BasePic, Mat &AddPic, int num);
Mat calcHomo2(Mat &BasePic, Mat &AddPic, int num);
void makePanorama1();

//点(x,y)がホモグラフィ行列homoによって変換されるx座標を返す
int calchomo_y(Mat homo, int x, int y)
{
	return (homo.at<double>(1, 0) * x + homo.at<double>(1, 1) * y + homo.at<double>(1, 2))
		/ (homo.at<double>(2, 0) * x + homo.at<double>(2, 1) * y + homo.at<double>(2, 2));
}

//点(x,y)がホモグラフィ行列homoによって変換されるy座標を返す
int calchomo_x(Mat homo, int x, int y)
{
	return (homo.at<double>(0, 0) * x + homo.at<double>(0, 1) * y + homo.at<double>(0, 2))
		/ (homo.at<double>(2, 0) * x + homo.at<double>(2, 1) * y + homo.at<double>(2, 2));
}

//homoを使ってBasePicをホモグラフィ変換しresultに対して合成を行う。
Mat homography(Mat homo, Mat BasePic, Mat result, int k)
{
	/////////////////////////////////////
	///ホモグラフィ変換//////////////////
	/////////////////////////////////////

	//合成に必要な余白
	int xmin = 0;
	int xmax = result.cols - 1;
	int ymin = 0;
	int ymax = result.rows - 1;

	//画像サイズを調べる。合成に必要な余白を調べる。
	int corner[4][2];  //四隅 [0]:x [1]:y
	corner[0][0] = calchomo_x(homo, 0, 0);
	corner[0][1] = calchomo_y(homo, 0, 0);
	corner[1][0] = calchomo_x(homo, BasePic.cols, 0);
	corner[1][1] = calchomo_y(homo, BasePic.cols, 0);
	corner[2][0] = calchomo_x(homo, BasePic.cols, BasePic.rows);
	corner[2][1] = calchomo_y(homo, BasePic.cols, BasePic.rows);
	corner[3][0] = calchomo_x(homo, 0, BasePic.rows);
	corner[3][1] = calchomo_y(homo, 0, BasePic.rows);
	for (int i = 0; i < 4; i++)
	{
		if (ymin > corner[i][1])
		{
			ymin = corner[i][1];
		}
		if (ymax < corner[i][1])
		{
			ymax = corner[i][1];
		}
		if (xmin > corner[i][0])
		{
			xmin = corner[i][0];
		}
		if (xmax < corner[i][0])
		{
			xmax = corner[i][0];
		}
	}
	/*
	for (int y = 0; y < BasePic.rows; y++){
	for (int x = 0; x < BasePic.cols; x++){
	int ytemp = (homo.at<double>(1, 0) * x + homo.at<double>(1, 1) * y + homo.at<double>(1, 2))
	/ (homo.at<double>(2, 0) * x + homo.at<double>(2, 1) * y + homo.at<double>(2, 2));
	int xtemp = (homo.at<double>(0, 0) * x + homo.at<double>(0, 1) * y + homo.at<double>(0, 2))
	/ (homo.at<double>(2, 0) * x + homo.at<double>(2, 1) * y + homo.at<double>(2, 2));
	if (ymin > ytemp)
	{
	ymin = ytemp;
	}
	else if (ymax < ytemp)
	{
	ymax = ytemp;
	}
	else if (xmin > xtemp)
	{
	xmin = xtemp;
	}
	else if (xmax < xtemp)
	{
	xmax = xtemp;
	}
	}
	}
	*/

	//合成をするためにresultのサイズを変更する。
	cv::Mat result2 = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC3);

	for (int y = 0; y < result.rows; y++){
		for (int x = 0; x < result.cols; x++){
			result2.at<Vec3b>(y - ymin,x - xmin) = result.at<Vec3b>(y, x);
		}
	}
	result.release();

	//ホモグラフィ変換した結果をhomopicに格納
	cv::Mat homopic = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC3);

	//補間すべきピクセルの判定を行う。indexが0なら要補間、1なら補間の必要なし
	Index index;
	for (int y = 0; y < homopic.rows; y++)
	{
		vector<int> temp;
		for (int x = 0; x < homopic.cols; x++)
		{
			temp.push_back(0);
		}
		index.push_back(temp);
	}

	//変換した結果をhomopicに格納
	for (int y = 0; y < BasePic.rows; y++){
		for (int x = 0; x < BasePic.cols; x++){
			int ytemp = (homo.at<double>(1, 0) * x + homo.at<double>(1, 1) * y + homo.at<double>(1, 2))
				/ (homo.at<double>(2, 0) * x + homo.at<double>(2, 1) * y + homo.at<double>(2, 2));
			int xtemp = (homo.at<double>(0, 0) * x + homo.at<double>(0, 1) * y + homo.at<double>(0, 2))
				/ (homo.at<double>(2, 0) * x + homo.at<double>(2, 1) * y + homo.at<double>(2, 2));
			for (int i = 0; i < 3; i++)
			{
				homopic.at<Vec3b>(ytemp - ymin, xtemp - xmin)[i] = BasePic.at<Vec3b>(y, x)[i];
			}
			index.at(ytemp - ymin).at(xtemp - xmin) = 1;
		}
	}

	//四隅の座標を更新する。
	for (int i = 0; i < 4; i++)
	{
		corner[i][0] = corner[i][0] - xmin;
		corner[i][1] = corner[i][1] - ymin;
	}
	
	//補間用にエッジの直線を求める。 //a[]:傾き　b[]:切片 , y = ax + b
	double a[4];	
	double b[4];
	
	for (int i = 0; i < 4; i++)
	{
		a[i] = ((double)(corner[(i + 1) % 4][1] - corner[i][1])) / (corner[(i + 1) % 4][0] - corner[i][0]);
		b[i] = corner[i][1] - (a[i] * corner[i][0]);
	}

	//左右については x = (y - b)/aの形に変換
	a[1] = 1 / a[1];
	a[3] = 1 / a[3];
	b[1] = -b[1] * a[1];
	b[3] = -b[3] * a[3];
		 
	//線形補間
	for (int y = 1; y < homopic.rows - 1; y++)
	{
		for (int x = 1; x < homopic.cols - 1; x++)
		{
			//indexが0　かつ　[0]より下,[1]より左,[2]より上,[3]より右
			if (index.at(y).at(x) == 0 && y > a[0] * x + b[0] && x < a[1] * y + b[1] && y < a[2] * x + b[2] && x > a[3] * y + b[3])
			{
				for (int i = 0; i < 3; i++)
				{
					int temp = 0;
					int count = 0;
					for (int a = 0; a < 3; a++)
					{
						for (int b = 0; b < 3; b++)
						{
							if (homopic.at<Vec3b>(y - 1 + a, x - 1 + b)[i] != 0)
							{
								temp += homopic.at<Vec3b>(y - 1 + a, x - 1 + b)[i];
								count++;
							}
						}
					}
					if (count == 0)
					{
						//std::cout << "0ばっか : " << x << "," << y << std::endl;
					}
					else
					{
						homopic.at<Vec3b>(y, x)[i] = temp / count;
					}
				}
			}
		}
	}


	resize(homopic, homopic, homopic.size());

	imwrite(HomeDir + "/ransachomoammae" + std::to_string(k) + ".jpg", homopic);

	//画像の合成
	mix2Pic(result2, homopic, k);

	imwrite(HomeDir + "/ransacammae" + std::to_string(k) + ".jpg", result2);

	std::cout << k << "完了" << std::endl;


	return result2;
}

//input10をベースにinput20を変換し合成を行う。特徴点をRANSACにより絞る
Mat matching2(Mat input1, Mat input2, int k)
{
	Mat result;  //合成画像を格納
	Mat homopic; //AddPic をホモグラフィ変換した後の画像を格納
	Mat img_matches; //2画像のマッチング結果を格納

	/////////////////////////////////////////////////////////////
	////////(1)sift特徴を用いてマッチング				/////////
	/////////////////////////////////////////////////////////////
	Mat gray[2];
	cv::cvtColor(input1, gray[0], CV_BGR2GRAY);
	cv::cvtColor(input2, gray[1], CV_BGR2GRAY);

	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extrator;

	vector<cv::KeyPoint> keypoints1, keypoints2;
	Mat descriptors[2];

	detector.detect(gray[0], keypoints1);
	extrator.compute(gray[0], keypoints1, descriptors[0]);

	detector.detect(gray[1], keypoints2);
	extrator.compute(gray[1], keypoints2, descriptors[1]);

	FlannBasedMatcher matcher;

	vector<DMatch> matches;
	matcher.match(descriptors[0], descriptors[1], matches);

	/////////////////////////////////////////////////////////////
	//////(2)RANSACを用いてマッチング結果からoutlierを検出///////
	/////////////////////////////////////////////////////////////
	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = keypoints1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	Mat H;
	vector<uchar> RANSACStatus;
	H = findFundamentalMat(p1, p2, RANSACStatus, FM_RANSAC, 3.0, 0.99);

	int OutlinerCount = 0;
	for (int i = 0; i<ptCount; i++)
	{
		if (RANSACStatus[i] == 0) // 0:outliner
		{
			OutlinerCount++;
		}
	}

	// calculate inliner
	vector<Point2f> Inlier1;
	vector<Point2f> Inlier2;
	vector<DMatch> Inliermatches;

	int InlinerCount = ptCount - OutlinerCount;
	Inliermatches.resize(InlinerCount);
	Inlier1.resize(InlinerCount);
	Inlier2.resize(InlinerCount);
	InlinerCount = 0;
	for (int i = 0; i<ptCount; i++)
	{
		if (RANSACStatus[i] != 0)
		{
			Inlier1[InlinerCount].x = p1.at<float>(i, 0);
			Inlier1[InlinerCount].y = p1.at<float>(i, 1);
			Inlier2[InlinerCount].x = p2.at<float>(i, 0);
			Inlier2[InlinerCount].y = p2.at<float>(i, 1);
			Inliermatches[InlinerCount].queryIdx = InlinerCount;
			Inliermatches[InlinerCount].trainIdx = InlinerCount;
			InlinerCount++;
		}
	}


	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(Inlier1, key1);
	KeyPoint::convert(Inlier2, key2);

	//output
	drawMatches(input1, key1, input2, key2, Inliermatches, img_matches);
	//imshow("matches",img_matches);

	imwrite(HomeDir + "/ransac" + std::to_string(k) + "a.jpg", img_matches);

	///////////////////////////////////////////////////////////////
	///(3)RANSACの結果からホモグラフィを計算					///
	///////////////////////////////////////////////////////////////

	//Mat matchedImg;

	vector<cv::Vec2f> points1(Inliermatches.size());
	vector<cv::Vec2f> points2(Inliermatches.size());

	for (size_t i = 0; i < Inliermatches.size(); ++i)
	{
		points1[i][0] = key1[Inliermatches[i].queryIdx].pt.x;
		points1[i][1] = key1[Inliermatches[i].queryIdx].pt.y;

		points2[i][0] = key2[Inliermatches[i].trainIdx].pt.x;
		points2[i][1] = key2[Inliermatches[i].trainIdx].pt.y;
	}

	//drawMatches(input1, key1, input2, key2, Inliermatches, matchedImg);


	//imwrite(HomeDir + "/ransac" + std::to_string(k) + "b.jpg", matchedImg);

	Mat homo = cv::findHomography(points2, points1, CV_RANSAC);


	//////////////////////////////////////////////////////////////////////
	//(4)ホモグラフィの計算結果からinput2を変形し、ベース画像に貼り付け///
	//////////////////////////////////////////////////////////////////////

	result = input1;

	result = homography(homo, input2, result, k);

	return result;

}

//input1とinput2のsift特徴のマッチング結果を返す。
matKey FindKeyPoint(Mat input1, Mat input2)
{
	matKey result;

	/////////////////////////////////////////////////////////////
	////////(1)sift特徴を用いてマッチング				/////////
	/////////////////////////////////////////////////////////////
	Mat gray[2];
	cv::cvtColor(input1, gray[0], CV_BGR2GRAY);
	cv::cvtColor(input2, gray[1], CV_BGR2GRAY);

	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extrator;

	vector<cv::KeyPoint> keypoints1, keypoints2;
	Mat descriptors[2];

	detector.detect(gray[0], keypoints1);
	extrator.compute(gray[0], keypoints1, descriptors[0]);

	detector.detect(gray[1], keypoints2);
	extrator.compute(gray[1], keypoints2, descriptors[1]);

	FlannBasedMatcher matcher;

	vector<DMatch> matches;
	matcher.match(descriptors[0], descriptors[1], matches);

	/////////////////////////////////////////////////////////////
	//////(2)RANSACを用いてマッチング結果からoutlierを検出///////
	/////////////////////////////////////////////////////////////
	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = keypoints1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	Mat H;
	vector<uchar> RANSACStatus;
	H = findFundamentalMat(p1, p2, RANSACStatus, FM_RANSAC, 3.0, 0.99);

	int OutlinerCount = 0;
	for (int i = 0; i<ptCount; i++)
	{
		if (RANSACStatus[i] == 0) // 0:outliner
		{
			OutlinerCount++;
		}
	}

	// calculate inliner
	vector<Point2f> Inlier1;
	vector<Point2f> Inlier2;
	//vector<DMatch> Inliermatches;

	int InlinerCount = ptCount - OutlinerCount;
	//Inliermatches.resize(InlinerCount);
	Inlier1.resize(InlinerCount);
	Inlier2.resize(InlinerCount);
	InlinerCount = 0;
	for (int i = 0; i<ptCount; i++)
	{
		if (RANSACStatus[i] != 0)
		{
			Inlier1[InlinerCount].x = (int)p1.at<float>(i, 0);
			Inlier1[InlinerCount].y = (int)p1.at<float>(i, 1);
			Inlier2[InlinerCount].x = (int)p2.at<float>(i, 0);
			Inlier2[InlinerCount].y = (int)p2.at<float>(i, 1);
			//Inliermatches[InlinerCount].queryIdx = InlinerCount;
			//Inliermatches[InlinerCount].trainIdx = InlinerCount;
			InlinerCount++;
		}
	}


	//vector<KeyPoint> key1(InlinerCount);
	//vector<KeyPoint> key2(InlinerCount);
	//KeyPoint::convert(Inlier1, key1);
	//KeyPoint::convert(Inlier2, key2);

	result.push_back(Inlier1);
	result.push_back(Inlier2);

	return result;
}

//外側の余白部分を消す。
Mat marginCut(Mat src)
{
	// 余白の開始、終了地点をチェック
	int bhs = 0;	
	int bhe = src.rows;
	int bws = 0;	
	int bwe = src.cols;	

	bool margin = true;
	
	Vec3b zero = (0, 0, 0);

	// x方向のleft側の余白をとる
	for (int i = 0; i < src.cols; i++)
	{
		for (int j = 0; j < src.rows; j++)
		{
			if (src.at<Vec3b>(j, i) != zero)
			{
				margin = false;
			}
		}
		if (margin)
		{
			bws = i;
		}
		else
		{
			margin = true;
			break;
		}
	}

	// x方向のright側の余白をとる
	for (int i = src.cols - 1; i >= 0; i--)
	{
		for (int j = 0; j < src.rows; j++)
		{
			if (src.at<Vec3b>(j, i) != zero)
			{
				margin = false;
			}
		}
		if (margin)
		{
			bwe = i;
		}
		else
		{
			margin = true;
			break;
		}
	}

	// y方向のtop側の余白をとる
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j) != zero)
			{
				margin = false;
			}
		}
		if (margin)
		{
			bhs = i;
		}
		else
		{
			margin = true;
			break;
		}
	}
	
	// y方向のbottom側の余白をとる
	for (int i = src.rows - 1; i >= 0; i--)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j) != zero)
			{
				margin = false;
			}
		}
		if (margin)
		{
			bhe = i;
		}
		else
		{
			margin = true;
			break;
		}
	}


	cv::Mat result = cv::Mat::zeros(bhe - bhs - 1 , bwe - bws - 1, CV_8UC3);
	src(cv::Rect(bws + 1, bhs + 1, bwe - bws - 1, bhe - bhs - 1)).copyTo(result);

	return result;
}

//BasePicにAddPicを合成する。
void mix2Pic(Mat &BasePic, Mat &AddPic, int k)
{
	Vec3b zero = (0, 0, 0);
	for (int y = 0; y < AddPic.rows; y++){
		for (int x = 0; x < AddPic.cols; x++){
			if (AddPic.at<Vec3b>(y, x) != zero)
			{
				if (BasePic.at<Vec3b>(y, x) == zero)
				{
					BasePic.at<Vec3b>(y, x) = AddPic.at<Vec3b>(y, x);
				}
				else
				{
					BasePic.at<Vec3b>(y, x)[0] = (BasePic.at<Vec3b>(y, x)[0] * k + AddPic.at<Vec3b>(y, x)[0]) / (k + 1);
					BasePic.at<Vec3b>(y, x)[1] = (BasePic.at<Vec3b>(y, x)[1] * k + AddPic.at<Vec3b>(y, x)[1]) / (k + 1);
					BasePic.at<Vec3b>(y, x)[2] = (BasePic.at<Vec3b>(y, x)[2] * k + AddPic.at<Vec3b>(y, x)[2]) / (k + 1);
				}
			}
		}
	}
}

//src画像をcenterを中心に拡大率rateだけズームしてdstに出力する。
void zoomPicture(cv::Mat src, cv::Mat dst, cv::Point2i center, double rate)
{
	if (rate < 1.0){//縮小は未対応なのでそのまま
		src.copyTo(dst);
		return;
	}

	cv::Mat resizeSrc;
	cv::resize(src, resizeSrc, cv::Size2i(0, 0), rate, rate);
	//拡大後の拡大中心
	cv::Point2i resizeCenter(center.x*rate, center.y*rate);

	//拡大中心と拡大率の設定次第で元の画像をはみ出してしまうので余白を入れる
	int blankHeight = src.rows / 2;//元画像の上下にそれぞれ入れる余白の画素数
	int blankWidth = src.cols / 2;//元画像の左右にそれぞれ入れる余白の画素数
	cv::Mat resizeSrcOnBlank = cv::Mat::zeros(resizeSrc.rows + 2 * blankHeight, resizeSrc.cols + 2 * blankWidth, CV_8UC3);
	resizeSrc.copyTo(resizeSrcOnBlank(cv::Rect(blankWidth, blankHeight, resizeSrc.cols, resizeSrc.rows)));
	resizeSrcOnBlank(cv::Rect(resizeCenter.x + blankWidth - src.cols / 2, resizeCenter.y + blankHeight - src.rows / 2, src.cols, src.rows)).copyTo(dst);
	return;

}

//src画像を外側に余白をspaceだけ作成して出力する。
Mat makePicture(cv::Mat src, double space)
{
	//余白を入れる
	int blankHeight = src.rows * space / 2;//元画像の上下にそれぞれ入れる余白の画素数
	int blankWidth = src.cols * space / 2;//元画像の左右にそれぞれ入れる余白の画素数

	cv::Mat dst = cv::Mat::zeros(src.rows + 2 * blankHeight, src.cols + 2 * blankWidth, CV_8UC3);
	src.copyTo(dst(cv::Rect(blankWidth, blankHeight, src.cols, src.rows)));

	//imshow("", dst);
	return dst;
}

//input10をベースにinput20を変換し合成を行う。特徴点をRANSACにより絞る
Mat matching(Mat input10, Mat input20, int k)
{
	Mat result;  //合成画像を格納
	Mat homopic; //AddPic をホモグラフィ変換した後の画像を格納
	Mat img_matches; //2画像のマッチング結果を格納

	//input に対して外側に余白を作った画像を作る。
	Mat input1;
	Mat input2;

	input1 = makePicture(input10, p-1);
	input2 = makePicture(input20, p-1);

	/////////////////////////////////////////////////////////////
	////////(1)sift特徴を用いてマッチング				/////////
	/////////////////////////////////////////////////////////////
	Mat gray[2];
	cv::cvtColor(input1, gray[0], CV_BGR2GRAY);
	cv::cvtColor(input2, gray[1], CV_BGR2GRAY);

	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extrator;

	vector<cv::KeyPoint> keypoints1, keypoints2;
	Mat descriptors[2];

	detector.detect(gray[0], keypoints1);
	extrator.compute(gray[0], keypoints1, descriptors[0]);
	
	detector.detect(gray[1], keypoints2);
	extrator.compute(gray[1], keypoints2, descriptors[1]);

	FlannBasedMatcher matcher;

	vector<DMatch> matches;
	matcher.match(descriptors[0], descriptors[1], matches);

	/////////////////////////////////////////////////////////////
	//////(2)RANSACを用いてマッチング結果からoutlierを検出///////
	/////////////////////////////////////////////////////////////
	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = keypoints1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	Mat H;
	vector<uchar> RANSACStatus;
	H = findFundamentalMat(p1, p2, RANSACStatus, FM_RANSAC, 3.0, 0.99);

	int OutlinerCount = 0;
	for (int i = 0; i<ptCount; i++)
	{
		if (RANSACStatus[i] == 0) // 0:outliner
		{
			OutlinerCount++;
		}
	}

	// calculate inliner
	vector<Point2f> Inlier1;
	vector<Point2f> Inlier2;
	vector<DMatch> Inliermatches;
	
	int InlinerCount = ptCount - OutlinerCount;
	Inliermatches.resize(InlinerCount);
	Inlier1.resize(InlinerCount);
	Inlier2.resize(InlinerCount);
	InlinerCount = 0;
	for (int i = 0; i<ptCount; i++)
	{
		if (RANSACStatus[i] != 0)
		{
			Inlier1[InlinerCount].x = p1.at<float>(i, 0);
			Inlier1[InlinerCount].y = p1.at<float>(i, 1);
			Inlier2[InlinerCount].x = p2.at<float>(i, 0);
			Inlier2[InlinerCount].y = p2.at<float>(i, 1);
			Inliermatches[InlinerCount].queryIdx = InlinerCount;
			Inliermatches[InlinerCount].trainIdx = InlinerCount;
			InlinerCount++;
		}
	}

	
	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(Inlier1, key1);
	KeyPoint::convert(Inlier2, key2);

	//output
	drawMatches(input1, key1, input2, key2, Inliermatches, img_matches);
	//imshow("matches",img_matches);

	imwrite(HomeDir + "/ransac" + std::to_string(k) + "a.jpg", img_matches);

	///////////////////////////////////////////////////////////////
	///(3)RANSACの結果からホモグラフィを計算					///
	///////////////////////////////////////////////////////////////

	//Mat matchedImg;

	vector<cv::Vec2f> points1(Inliermatches.size());
	vector<cv::Vec2f> points2(Inliermatches.size());

	for (size_t i = 0; i < Inliermatches.size(); ++i)
	{
		points1[i][0] = key1[Inliermatches[i].queryIdx].pt.x;
		points1[i][1] = key1[Inliermatches[i].queryIdx].pt.y;

		points2[i][0] = key2[Inliermatches[i].trainIdx].pt.x;
		points2[i][1] = key2[Inliermatches[i].trainIdx].pt.y;
	}

	//drawMatches(input1, key1, input2, key2, Inliermatches, matchedImg);


	//imwrite(HomeDir + "/ransac" + std::to_string(k) + "b.jpg", matchedImg);

	Mat homo = cv::findHomography(points2, points1, CV_RANSAC);


	//////////////////////////////////////////////////////////////////////
	//(4)ホモグラフィの計算結果からinput2を変形し、ベース画像に貼り付け///
	//////////////////////////////////////////////////////////////////////

	result = input1;

	//AddPic をホモグラフィ行列を用いて変形し、 homopic に格納する。
	//ー＞合成後画像の大きさは合成前画像の画素のあるサイズの1.3倍にする
	cv::warpPerspective(input2, homopic, homo, Size(static_cast<int>(result.cols), static_cast<int>(result.rows)));
	imwrite(HomeDir + "/ransachomo" + std::to_string(k) + ".jpg", homopic);

	//画像の合成
	mix2Pic(result, homopic, k);

	//余白を消す
	result = marginCut(result);

	imwrite(HomeDir + "/ransaclast" + std::to_string(k) + ".jpg", result);

	std::cout << k << "完了" << std::endl;

	return result;

}

//BasePic1をベースにAddPic1を変換し合成を行う。特徴点はそのまま使う
Mat stitching2Pic(Mat &BasePic1, Mat &AddPic1, int k)
{
	Mat gray[2];
	Mat result;  //合成画像を格納
	Mat homopic; //AddPic をホモグラフィ変換した後の画像を格納
	Mat BasePic;
	Mat AddPic;

	BasePic = makePicture(BasePic1, p-1);
	AddPic = makePicture(AddPic1, p-1);

	imwrite(HomeDir + "/test" + std::to_string(0) + ".jpg", BasePic);
	//imshow("", BasePic);
	//waitKey();


	cv::cvtColor(BasePic, gray[0], CV_BGR2GRAY);
	cv::cvtColor(AddPic, gray[1], CV_BGR2GRAY);

	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extrator;

	vector<cv::KeyPoint> keypoints[2];
	Mat descriptors[2];

	for (int i = 0; i < 2; i++){
		detector.detect(gray[i], keypoints[i]);
		extrator.compute(gray[i], keypoints[i], descriptors[i]);
	}

	vector<cv::DMatch> matches;
	cv::BruteForceMatcher< cv::L2<float> > matcher;
	matcher.match(descriptors[0], descriptors[1], matches);

	vector<cv::Vec2f> points1(matches.size());
	vector<cv::Vec2f> points2(matches.size());

	for (size_t i = 0; i < matches.size(); ++i)
	{
		points1[i][0] = keypoints[0][matches[i].queryIdx].pt.x;
		points1[i][1] = keypoints[0][matches[i].queryIdx].pt.y;

		points2[i][0] = keypoints[1][matches[i].trainIdx].pt.x;
		points2[i][1] = keypoints[1][matches[i].trainIdx].pt.y;
	}

	Mat matchedImg;
	drawMatches(BasePic, keypoints[0], AddPic, keypoints[1], matches, matchedImg);
	//imshow("draw img", matchedImg);
	//waitKey();
	imwrite(HomeDir + "/matching" + std::to_string(k) + ".jpg", matchedImg);


	//points2 から points1 への変換を行うためのホモグラフィの計算を行う
	Mat homo = cv::findHomography(points2, points1, CV_RANSAC);


	result = BasePic;

	//AddPic をホモグラフィ行列を用いて変形し、 homopic に格納する。
	cv::warpPerspective(AddPic, homopic, homo, Size(static_cast<int>(result.cols), static_cast<int>(result.rows)));
	imwrite(HomeDir + "/homo" + std::to_string(k) + ".jpg", homopic);	


	mix2Pic(result, BasePic, k);

	result = marginCut(result);

	//Vec3b zero = (0, 0, 0);

	//for (int y = 0; y < homopic.rows; y++){
	//	for (int x = 0; x < homopic.cols; x++){
	//		if (homopic.at<Vec3b>(y, x) != zero)
	//		{
	//			if (result.at<Vec3b>(y, x) == zero)
	//			{
	//				result.at<Vec3b>(y, x) = homopic.at<Vec3b>(y, x);
	//			}
	//			else
	//			{
	//				result.at<Vec3b>(y, x)[0] = (result.at<Vec3b>(y, x)[0] + homopic.at<Vec3b>(y, x)[0]) / 2;
	//				result.at<Vec3b>(y, x)[1] = (result.at<Vec3b>(y, x)[1] + homopic.at<Vec3b>(y, x)[1]) / 2;
	//				result.at<Vec3b>(y, x)[2] = (result.at<Vec3b>(y, x)[2] + homopic.at<Vec3b>(y, x)[2]) / 2;
	//			}
	//		}
	//	}
	//}

	//imshow("result img", result);
	//waitKey();

	imwrite(HomeDir + "/panorama" + std::to_string(k) + ".jpg", result);

	std::cout << "stitching完了" << std::endl;

	return result;
}

//BasePicをベースにAddPicを変換するhomographyを計算。numが指定された(0でない)ならばマッチング結果をcalc"num".jpgで保存。
Mat calcHomo(Mat &BasePic, Mat &AddPic, int num = 0)
{
	Mat gray[2];

	cv::cvtColor(BasePic, gray[0], CV_BGR2GRAY);
	cv::cvtColor(AddPic, gray[1], CV_BGR2GRAY);

	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extrator;

	vector<cv::KeyPoint> keypoints[2];
	Mat descriptors[2];

	for (int i = 0; i < 2; i++){
		detector.detect(gray[i], keypoints[i]);
		extrator.compute(gray[i], keypoints[i], descriptors[i]);
	}

	vector<cv::DMatch> matches;
	cv::BruteForceMatcher< cv::L2<float> > matcher;
	matcher.match(descriptors[0], descriptors[1], matches);

	vector<cv::Vec2f> points1(matches.size());
	vector<cv::Vec2f> points2(matches.size());

	for (size_t i = 0; i < matches.size(); ++i)
	{
		points1[i][0] = keypoints[0][matches[i].queryIdx].pt.x;
		points1[i][1] = keypoints[0][matches[i].queryIdx].pt.y;

		points2[i][0] = keypoints[1][matches[i].trainIdx].pt.x;
		points2[i][1] = keypoints[1][matches[i].trainIdx].pt.y;
	}

	//////////////////////////////////////////////////////////
	////////////////////マッチングの確認//////////////////////
	//////////////////////////////////////////////////////////
	if (num)
	{
		Mat matchedImg;
		drawMatches(BasePic, keypoints[0], AddPic, keypoints[1], matches, matchedImg);
		imwrite(HomeDir + "/calc" + std::to_string(num) + ".jpg", matchedImg);
	}

	//points2 から points1 への変換を行うためのホモグラフィの計算を行う
	Mat homo = cv::findHomography(points2, points1, CV_RANSAC);

	return homo;
}

//BasePicをベースにAddPicを変換するhomographyを計算。numが指定された(0でない)ならばマッチング結果をcalc"num".jpgで保存。RANSAC使用
Mat calcHomo2(Mat &BasePic, Mat &AddPic, int num = 0)
{
	/////////////////////////////////////////////////////////////
	////////(1)sift特徴を用いてマッチング				/////////
	/////////////////////////////////////////////////////////////

	Mat gray[2];
	cv::cvtColor(BasePic, gray[0], CV_BGR2GRAY);
	cv::cvtColor(AddPic, gray[1], CV_BGR2GRAY);

	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extrator;

	vector<cv::KeyPoint> keypoints1, keypoints2;
	Mat descriptors[2];

	detector.detect(gray[0], keypoints1);
	extrator.compute(gray[0], keypoints1, descriptors[0]);

	detector.detect(gray[1], keypoints2);
	extrator.compute(gray[1], keypoints2, descriptors[1]);

	FlannBasedMatcher matcher;

	vector<DMatch> matches;
	matcher.match(descriptors[0], descriptors[1], matches);

	/////////////////////////////////////////////////////////////
	//////(2)RANSACを用いてマッチング結果からoutlierを検出///////
	/////////////////////////////////////////////////////////////
	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = keypoints1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	Mat H;
	vector<uchar> RANSACStatus;
	H = findFundamentalMat(p1, p2, RANSACStatus, FM_RANSAC, 3.0, 0.99);

	int OutlinerCount = 0;
	for (int i = 0; i<ptCount; i++)
	{
		if (RANSACStatus[i] == 0) // 0:outliner
		{
			OutlinerCount++;
		}
	}

	// calculate inliner
	vector<Point2f> Inlier1;
	vector<Point2f> Inlier2;
	vector<DMatch> Inliermatches;

	int InlinerCount = ptCount - OutlinerCount;
	Inliermatches.resize(InlinerCount);
	Inlier1.resize(InlinerCount);
	Inlier2.resize(InlinerCount);
	InlinerCount = 0;
	for (int i = 0; i<ptCount; i++)
	{
		if (RANSACStatus[i] != 0)
		{
			Inlier1[InlinerCount].x = p1.at<float>(i, 0);
			Inlier1[InlinerCount].y = p1.at<float>(i, 1);
			Inlier2[InlinerCount].x = p2.at<float>(i, 0);
			Inlier2[InlinerCount].y = p2.at<float>(i, 1);
			Inliermatches[InlinerCount].queryIdx = InlinerCount;
			Inliermatches[InlinerCount].trainIdx = InlinerCount;
			InlinerCount++;
		}
	}

	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(Inlier1, key1);
	KeyPoint::convert(Inlier2, key2);

	/////////////////////////////////////////////////////////////////
	///(3)RANSACの結果から再度マッチングを行い、ホモグラフィの計算///
	/////////////////////////////////////////////////////////////////

	cv::BruteForceMatcher< cv::L2<float> > matcher2;
	Mat descriptors2[2];
	vector<cv::DMatch> matches2;

	extrator.compute(gray[0], key1, descriptors2[0]);
	extrator.compute(gray[1], key2, descriptors2[1]);

	matcher2.match(descriptors2[0], descriptors2[1], matches2);

	vector<cv::Vec2f> points1(matches2.size());
	vector<cv::Vec2f> points2(matches2.size());

	for (size_t i = 0; i < matches2.size(); ++i)
	{
		points1[i][0] = key1[matches2[i].queryIdx].pt.x;
		points1[i][1] = key1[matches2[i].queryIdx].pt.y;

		points2[i][0] = key2[matches2[i].trainIdx].pt.x;
		points2[i][1] = key2[matches2[i].trainIdx].pt.y;
	}

	//points2 から points1 への変換を行うためのホモグラフィの計算を行う
	Mat homo = cv::findHomography(points2, points1, CV_RANSAC);

	//////////////////////////////////////////////////////////
	////////////////////マッチングの確認//////////////////////
	//////////////////////////////////////////////////////////
	if (num)
	{
		Mat matchedImg;
		drawMatches(BasePic, key1, AddPic, key2, matches2, matchedImg);
		imwrite(HomeDir + "/calc" + std::to_string(num) + ".jpg", matchedImg);
	}

	return homo;
}

//EigenのSparseMatを用いてSVDによってPoint[1]からPoint[0]へのホモグラフィ行列を計算し返す
Mat FindHomographyWithSVD_EigenSparse(matKey Point)
{
	int num = Point[0].size();  //Matching の個数
	int numPic = Point.size();  //画像数
	int n = numPic * (numPic - 1) / 2;  //ホモグラフィ行列の個数

	//Eigen を利用して特異値分解を行う
	Eigen::SparseMatrix<int> A(2 * num, 9 * n);
	std::vector<T> v;

	//for (int i = 0; i < num; i++)
	//{
	//	v.push_back(T(2 * i, 3, Point[1][i].x));
	//	v.push_back(T(2 * i, 4, Point[1][i].y));
	//	v.push_back(T(2 * i, 5, 1));

	//	v.push_back(T(2 * i + 1, 0, Point[1][i].x));
	//	v.push_back(T(2 * i + 1, 1, Point[1][i].y));
	//	v.push_back(T(2 * i + 1, 2, 1));

	//	v.push_back(T(2 * i, 6, 0 - Point[0][i].y * Point[1][i].x));
	//	v.push_back(T(2 * i, 7, 0 - Point[0][i].y * Point[1][i].y));
	//	v.push_back(T(2 * i, 8, 0 - Point[0][i].y));

	//	v.push_back(T(2 * i + 1, 6, 0 - Point[0][i].x * Point[1][i].x));
	//	v.push_back(T(2 * i + 1, 7, 0 - Point[0][i].x * Point[1][i].y));
	//	v.push_back(T(2 * i + 1, 8, 0 - Point[0][i].x));
	//}

	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < n; j++)
		{
			v.push_back(T(2 * i, 3 + 9 * j, Point[(j + 1) % 3][i].x));
			v.push_back(T(2 * i, 4 + 9 * j, Point[(j + 1) % 3][i].y));
			v.push_back(T(2 * i, 5 + 9 * j, 1));

			v.push_back(T(2 * i + 1, 0 + 9 * j, Point[(j + 1) % 3][i].x));
			v.push_back(T(2 * i + 1, 1 + 9 * j, Point[(j + 1) % 3][i].y));
			v.push_back(T(2 * i + 1, 2 + 9 * j, 1));

			v.push_back(T(2 * i, 6 + 9 * j, 0 - Point[j][i].y * Point[(j + 1) % 3][i].x));
			v.push_back(T(2 * i, 7 + 9 * j, 0 - Point[j][i].y * Point[(j + 1) % 3][i].y));
			v.push_back(T(2 * i, 8 + 9 * j, 0 - Point[j][i].y));

			v.push_back(T(2 * i + 1, 6 + 9 * j, 0 - Point[j][i].x * Point[(j + 1) % 3][i].x));
			v.push_back(T(2 * i + 1, 7 + 9 * j, 0 - Point[j][i].x * Point[(j + 1) % 3][i].y));
			v.push_back(T(2 * i + 1, 8 + 9 * j, 0 - Point[j][i].x));
		}
	}

	A.setFromTriplets(v.begin(), v.end());
	//std::cout << std::endl << "FindHomographyWithSVD_EigenSparse:" << std::endl << A << std::endl << std::endl;

	JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);

	//MatrixXd S;
	MatrixXd V;
	//V = svd.matrixV();
	V = svd.matrixV().col(9 * n - 1);

	//std::cout << std::endl << "FindHomographyWithSVD_EigenSparse:" << std::endl << svd.matrixV() << std::endl;

	//std::cout << std::endl << "FindHomographyWithSVD_EigenSparse:" << std::endl << V << std::endl;

	vector<vector<double>> homolist;

	for (int i = 0; i < n; i++)
	{
		vector<double> homo;
		for (int j = 0; j < 9; j++)
		{
			homo.push_back(V(j + 9 * i, 0) / V(8 + 9 * i, 0));
		}
		homolist.push_back(homo);
	}

	//double homo[9];

	//homo[0] = V(0, 0) / V(8, 0);
	//homo[1] = V(1, 0) / V(8, 0);
	//homo[2] = V(2, 0) / V(8, 0);
	//homo[3] = V(3, 0) / V(8, 0);
	//homo[4] = V(4, 0) / V(8, 0);
	//homo[5] = V(5, 0) / V(8, 0);
	//homo[6] = V(6, 0) / V(8, 0);
	//homo[7] = V(7, 0) / V(8, 0);
	//homo[8] = V(8, 0) / V(8, 0);

	vector<Mat> Homolist;

	for (int i = 0; i < n; i++)
	{
		Mat Homo = (cv::Mat_<double>(3, 3) <<
			homolist.at(i).at(0), homolist.at(i).at(1), homolist.at(i).at(2),
			homolist.at(i).at(3), homolist.at(i).at(4), homolist.at(i).at(5),
			homolist.at(i).at(6), homolist.at(i).at(7), homolist.at(i).at(8));

		Homolist.push_back(Homo);
		std::cout << std::endl << "FindHomographyWithSVD_EigenSparse:" << std::endl << Homo << std::endl;
	}

	//Mat Homo = (cv::Mat_<double>(3, 3) << homo[0], homo[1], homo[2],
	//	homo[3], homo[4], homo[5],
	//	homo[6], homo[7], homo[8]);

	//std::cout << std::endl << "FindHomographyWithSVD_EigenSparse:" << std::endl << Homo << std::endl;

	Mat Homo = Homolist.at(0);

	return Homo;
}

//EigenのDenseMatを用いてSVDによってPoint[1]からPoint[0]へのホモグラフィ行列を計算し返す
Mat FindHomographyWithSVD_EigenDense(matKey Point)
{
	int num = Point[0].size();

	Eigen::MatrixXd A(2 * num, 9);

	for (int i = 0; i < num; i++)
	{
		A(2 * i, 0) = 0;
		A(2 * i, 1) = 0;
		A(2 * i, 2) = 0;

		A(2 * i, 3) = Point[1][i].x;
		A(2 * i, 4) = Point[1][i].y;
		A(2 * i, 5) = 1;

		A(2 * i + 1, 0) = Point[1][i].x;
		A(2 * i + 1, 1) = Point[1][i].y;
		A(2 * i + 1, 2) = 1;

		A(2 * i + 1, 3) = 0;
		A(2 * i + 1, 4) = 0;
		A(2 * i + 1, 5) = 0;

		A(2 * i, 6) = 0 - Point[0][i].y * Point[1][i].x;
		A(2 * i, 7) = 0 - Point[0][i].y * Point[1][i].y;
		A(2 * i, 8) = 0 - Point[0][i].y;

		A(2 * i + 1, 6) = 0 - Point[0][i].x * Point[1][i].x;
		A(2 * i + 1, 7) = 0 - Point[0][i].x * Point[1][i].y;
		A(2 * i + 1, 8) = 0 - Point[0][i].x;
	}
	//std::cout << std::endl << "FindHomographyWithSVD_EigenDense:" << std::endl << A << std::endl << std::endl;

	JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);

	//MatrixXd S;
	MatrixXd V;
	//V = svd.matrixV();
	V = svd.matrixV().col(8);

	//std::cout << std::endl << "FindHomographyWithSVD_EigenDense:" << std::endl << svd.matrixV() << std::endl;

	//std::cout << std::endl << "FindHomographyWithSVD_EigenDense:" << std::endl << V << std::endl;


	double homo[9];

	homo[0] = V(0, 0) / V(8, 0);
	homo[1] = V(1, 0) / V(8, 0);
	homo[2] = V(2, 0) / V(8, 0);
	homo[3] = V(3, 0) / V(8, 0);
	homo[4] = V(4, 0) / V(8, 0);
	homo[5] = V(5, 0) / V(8, 0);
	homo[6] = V(6, 0) / V(8, 0);
	homo[7] = V(7, 0) / V(8, 0);
	homo[8] = V(8, 0) / V(8, 0);

	Mat Homo = (cv::Mat_<double>(3, 3) << homo[0], homo[1], homo[2],
		homo[3], homo[4], homo[5],
		homo[6], homo[7], homo[8]);

	std::cout << std::endl << "FindHomographyWithSVD_EigenDense:" << std::endl << Homo << std::endl;

	return Homo;

}

//OpenCVのDenseMatを用いてSVDによってPoint[1]からPoint[0]へのホモグラフィ行列を計算し返す
Mat FindHomographyWithSVD_CVDense(matKey Point)
{
	int num = Point[0].size();
	Mat A = (cv::Mat_<double>(2 * num, 9));

	for (int i = 0; i < num; i++)
	{
		A.at<double>(2 * i, 0) = 0;
		A.at<double>(2 * i, 1) = 0;
		A.at<double>(2 * i, 2) = 0;

		A.at<double>(2 * i, 3) = Point[1][i].x;
		A.at<double>(2 * i, 4) = Point[1][i].y;
		A.at<double>(2 * i, 5) = 1;

		A.at<double>(2 * i + 1, 0) = Point[1][i].x;
		A.at<double>(2 * i + 1, 1) = Point[1][i].y;
		A.at<double>(2 * i + 1, 2) = 1;

		A.at<double>(2 * i + 1, 3) = 0;
		A.at<double>(2 * i + 1, 4) = 0;
		A.at<double>(2 * i + 1, 5) = 0;

		A.at<double>(2 * i, 6) = -Point[0][i].y * Point[1][i].x;
		A.at<double>(2 * i, 7) = -Point[0][i].y * Point[1][i].y;
		A.at<double>(2 * i, 8) = -Point[0][i].y;

		A.at<double>(2 * i + 1, 6) = -Point[0][i].x * Point[1][i].x;
		A.at<double>(2 * i + 1, 7) = -Point[0][i].x * Point[1][i].y;
		A.at<double>(2 * i + 1, 8) = -Point[0][i].x;
	}

	Mat Homotemp;
	SVD::solveZ(A, Homotemp);

	//std::cout << std::endl << "FindHomographyWithSVD_CVDense:" << std::endl << Homotemp << std::endl;


	Mat Homo = (cv::Mat_<double>(3, 3));


	Homo.at<double>(0, 0) = Homotemp.at<double>(0, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(0, 1) = Homotemp.at<double>(1, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(0, 2) = Homotemp.at<double>(2, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(1, 0) = Homotemp.at<double>(3, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(1, 1) = Homotemp.at<double>(4, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(1, 2) = Homotemp.at<double>(5, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(2, 0) = Homotemp.at<double>(6, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(2, 1) = Homotemp.at<double>(7, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(2, 2) = Homotemp.at<double>(8, 0) / Homotemp.at<double>(8, 0);


	std::cout << std::endl << "FindHomographyWithSVD_CVDense:" << std::endl << Homo << std::endl;

	return Homo;
}

//OpenCVのDenseMatを用いてLeast-SquareによってPoint[1]からPoint[0]へのホモグラフィ行列を計算し返す
Mat FindHomographyWithLS_CVDense(matKey Point)
{
	int num = Point[0].size();
	Mat A = (cv::Mat_<double>(3 * num, 9));
	Mat B = (cv::Mat_<double>(3 * num, 1));

	for (int i = 0; i < num; i++)
	{
		A.at<double>(3 * i, 0) = Point[1][i].x;
		A.at<double>(3 * i, 1) = Point[1][i].y;
		A.at<double>(3 * i, 2) = 1;
		A.at<double>(3 * i, 3) = 0;
		A.at<double>(3 * i, 4) = 0;
		A.at<double>(3 * i, 5) = 0;
		A.at<double>(3 * i, 6) = 0;
		A.at<double>(3 * i, 7) = 0;
		A.at<double>(3 * i, 8) = 0;

		A.at<double>(3 * i + 1, 0) = 0;
		A.at<double>(3 * i + 1, 1) = 0;
		A.at<double>(3 * i + 1, 2) = 0;
		A.at<double>(3 * i + 1, 3) = Point[1][i].x;
		A.at<double>(3 * i + 1, 4) = Point[1][i].y;
		A.at<double>(3 * i + 1, 5) = 1;
		A.at<double>(3 * i + 1, 6) = 0;
		A.at<double>(3 * i + 1, 7) = 0;
		A.at<double>(3 * i + 1, 8) = 0;

		A.at<double>(3 * i + 2, 0) = 0;
		A.at<double>(3 * i + 2, 1) = 0;
		A.at<double>(3 * i + 2, 2) = 0;
		A.at<double>(3 * i + 2, 3) = 0;
		A.at<double>(3 * i + 2, 4) = 0;
		A.at<double>(3 * i + 2, 5) = 0;
		A.at<double>(3 * i + 2, 6) = Point[1][i].x;
		A.at<double>(3 * i + 2, 7) = Point[1][i].y;
		A.at<double>(3 * i + 2, 8) = 1;

		B.at<double>(3 * i, 0) = Point[0][i].x;
		B.at<double>(3 * i + 1, 0) = Point[0][i].y;
		B.at<double>(3 * i + 2, 0) = 1;
	}

	Mat Homotemp;
	cv::solve(A, B, Homotemp, cv::DECOMP_SVD);

	//std::cout << std::endl << "FindHomographyWithLS_CVDense:" << std::endl << A << std::endl;
	//std::cout << std::endl << "FindHomographyWithLS_CVDense:" << std::endl << B << std::endl;


	//std::cout << std::endl << "FindHomographyWithLS_CVDense:" << std::endl << Homotemp << std::endl;


	Mat Homo = (cv::Mat_<double>(3, 3));

	Homo.at<double>(0, 0) = Homotemp.at<double>(0, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(0, 1) = Homotemp.at<double>(1, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(0, 2) = Homotemp.at<double>(2, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(1, 0) = Homotemp.at<double>(3, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(1, 1) = Homotemp.at<double>(4, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(1, 2) = Homotemp.at<double>(5, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(2, 0) = Homotemp.at<double>(6, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(2, 1) = Homotemp.at<double>(7, 0) / Homotemp.at<double>(8, 0);
	Homo.at<double>(2, 2) = Homotemp.at<double>(8, 0) / Homotemp.at<double>(8, 0);


	std::cout << std::endl << "FindHomographyWithLS_CVDense:" << std::endl << Homo << std::endl;

	return Homo;

}


void makePanorama1()
{
	Mat src[count];
	Mat tempresult;
	Mat result;
	
	for (int i = 0; i < count; i++)
	{
		src[i] = imread(HomeDir + std::to_string(i) + ".JPG");
	}

	if (dodo == 0)
	{
		tempresult = src[0];

		for (int i = 1; i < count; i++)
		{
			tempresult = matching2(tempresult, src[i], i);
			//tempresult = matching(tempresult, src[i], i);
		}

		result = tempresult;
	}
	else if (dodo == 1)
	{
		tempresult = src[0];

		for (int i = 1; i < count; i++)
		{
			tempresult = stitching2Pic(tempresult, src[i], i);
		}

		result = tempresult;
	}
	else if (dodo == 2)
	{
		matKey a, b, c;
		
		//for (int i = 1; i < count; i++)
		//{
		//	for (int j = 0; j < i; j++)
		//	{
		//		a = FindKeyPoint(src[i], src[j]);
		//	}
		//}
		a = FindKeyPoint(src[0], src[1]);
		b = FindKeyPoint(src[1], src[2]);
		c = FindKeyPoint(src[2], src[0]);
		int asize = a[0].size();
		int bsize = b[0].size();
		int csize = c[0].size();
		vector<Point2f> tempmatch;
		tempmatch.resize(asize);

		int matchnum = 0;
		vector<int> matchindex;
		bool matching = false;

		//0,1の結果に0,2の結果を追加した0,1,2を作る
		matchindex.resize(csize);
		for (int i = 0; i < asize; i++)
		{
			for (int j = 0; j < csize; j++)
			{
				if (a[0][i] == c[1][j])
				{
					tempmatch[i] = c[0][j];
					matchnum++;
					matchindex[j] = 1;
				}
			}
		}
		std::cout << matchnum << std::endl;
		a.push_back(tempmatch);

		for (int i = 0; i < csize; i++)
		{
			if (matchindex[i] == 0)
			{
				a[0].push_back(c[1][i]);
				a[2].push_back(c[0][i]);
			}
		}
		a[1].resize(a[0].size());


		//1,2だけの結果を0,1,2に加える
		asize = a[0].size();
		matchindex.clear();
		matching = false;
		for (int i = 0; i < bsize; i++)
		{
			for (int j = 0; j < asize; j++)
			{
				if (b[0][i] == a[1][j])
				{
					matching = true;
				}
			}
			if (!matching)
			{
				matchindex.push_back(i);
			}
			matching = false;
		}
		a[0].resize(asize + matchindex.size());
		a[1].resize(asize + matchindex.size());
		a[2].resize(asize + matchindex.size());

		matchnum = 0;
		for (int i = asize; i < a[0].size(); i++)
		{
			a[1][i] = b[0][matchindex[matchnum]];
			a[2][i] = b[1][matchindex[matchnum]];
			matchnum++;
		}

		Mat Homo = FindHomographyWithSVD_EigenSparse(a);

		Mat result;

		//result = homography(Homo,src[1],src[0],1);

		waitKey();
	}
	else if (dodo == 5)
	{

	}
	else if (dodo == 6)
	{
		matKey a;
		a = FindKeyPoint(src[0], src[1]);

		Mat Q, W, E, R;
		Q = FindHomographyWithLS_CVDense(a);
		W = FindHomographyWithSVD_CVDense(a);
		E = FindHomographyWithSVD_EigenDense(a);
		R = FindHomographyWithSVD_EigenSparse(a);

		Mat homotest = cv::findHomography(a[1], a[0], CV_RANSAC);

		std::cout << std::endl << "OpenCV_findHomography:" << std::endl << homotest << std::endl;

		result = homography(Q, src[1], src[0], 0);


		waitKey(0);

	}
	else if (dodo == 7)
	{
		int a, b;

		a = src[1].rows;
		b = src[1].cols;

		matKey nyu;
		vector<Point2f> temp, temp2;
		temp.resize(5);
		temp2.resize(5);

		temp[0].x = 106;
		temp[0].y = 108;
		temp[1].x = 796;
		temp[1].y = 0;
		temp[2].x = 812;
		temp[2].y = 512;
		temp[3].x = 133;
		temp[3].y = 487;

		temp[4].x = 278;
		temp[4].y = 82;

		nyu.push_back(temp);

		temp2[0].x = 0;
		temp2[0].y = 0;
		temp2[1].x = b;
		temp2[1].y = 0;
		temp2[2].x = b;
		temp2[2].y = a;
		temp2[3].x = 0;
		temp2[3].y = a;

		temp2[4].x = 198;
		temp2[4].y = 0;

		nyu.push_back(temp2);

		Mat Q, W, E, R;
		Q = FindHomographyWithLS_CVDense(nyu);
		W = FindHomographyWithSVD_CVDense(nyu);
		E = FindHomographyWithSVD_EigenDense(nyu);
		R = FindHomographyWithSVD_EigenSparse(nyu);

		Mat homotest = cv::findHomography(nyu[1],nyu[0], CV_RANSAC);
		std::cout << std::endl << "OpenCV_findHomography:" << std::endl << homotest << std::endl;

		waitKey(0);

	}
	else if (dodo == 4)  //もう使わない？
	{
		matKey a;

		a = FindKeyPoint(src[0], src[1]);

		Mat homotest = cv::findHomography(a[1], a[0], CV_RANSAC);
		std::cout << std::endl << "OpenCV_findHomography:" << std::endl << homotest << std::endl;

		Mat result;

		result = homography(homotest, src[1], src[0], 1);

		waitKey();

	}
	else if (dodo == 3)
	{
		Mat src2[count];

		for (int i = 0; i < count; i++)
		{
			src2[i] = makePicture(src[i], 1.5);
		}

		//for debug
		for (int i = 0; i < count; i++)
		{
			imwrite(HomeDir + "/pic" + std::to_string(i) + ".jpg", src2[i]);
		}

		tempresult.release();
		src->release();

		std::cout << "homo start" << std::endl;

		Mat homo21, homo31, homo32, homo23;

		homo21 = calcHomo2(src2[0], src2[1], 21);
		std::cout << "homo 21end" << std::endl;

		homo23 = calcHomo2(src2[2], src2[1], 23);
		std::cout << "homo 23end" << std::endl;

		homo31 = calcHomo2(src2[0], src2[2], 31);
		std::cout << "homo 31end" << std::endl;

		homo32 = calcHomo(src2[1], src2[2]);
		std::cout << "homo 32end" << std::endl;

		//計算開始

		Mat homopic21, homopic31;

		Mat homopic23, homopic231;

		//AddPic をホモグラフィ行列を用いて変形し、 homopic に格納する。
		cv::warpPerspective(src2[1], homopic21, homo21, Size(static_cast<int>(src2[1].cols), static_cast<int>(src2[1].rows)));
		imwrite(HomeDir + "/homo21.jpg", homopic21);
		cv::warpPerspective(src2[2], homopic31, homo31, Size(static_cast<int>(src2[2].cols), static_cast<int>(src2[2].rows)));
		imwrite(HomeDir + "/homo31.jpg", homopic31);

		//for debug

		//2枚目を3枚目に合成
		cv::warpPerspective(src2[1], homopic23, homo23, Size(static_cast<int>(src2[1].cols), static_cast<int>(src2[1].rows)));
		imwrite(HomeDir + "/homo23.jpg", homopic23);

		//Mat aaa = src2[2];
		//mix2Pic(aaa, homopic23);
		//imwrite(HomeDir + "/result23.jpg", aaa);

		////2,3枚目を1枚目に合成
		//cv::warpPerspective(homo23, homopic231, homo31, Size(static_cast<int>(homo23.cols), static_cast<int>(homo23.rows)));
		//imwrite(HomeDir + "/homo231.jpg", homopic231);

		//Mat aaaa = src2[0];
		//mix2Pic(aaaa, homopic231);
		//imwrite(HomeDir + "/result231.jpg", aaaa);

		//2枚目を3枚目の平面に射影し、1枚目の平面に射影
		Mat homopiclast;
		cv::warpPerspective(homopic23, homopiclast, homo31, Size(static_cast<int>(homopic23.cols), static_cast<int>(homopic23.rows)));
		imwrite(HomeDir + "/homo231.jpg", homopiclast);

		result = src2[0];

		mix2Pic(result, homopic21, 1);
		mix2Pic(result, homopic31, 1);
		
		imwrite(HomeDir + "/resultallto1.jpg", result);
		
		double miss = 0.0;
		for (int i = 0; i < homopiclast.cols; i++)
		{
			for (int j = 0; j < homopiclast.rows; j++)
			{
				miss += (homopic21.at<Vec3b>(j, i)[0] - homopiclast.at<Vec3b>(j, i)[0]) / 2;
				miss += (homopic21.at<Vec3b>(j, i)[1] - homopiclast.at<Vec3b>(j, i)[1]) / 2;
				miss += (homopic21.at<Vec3b>(j, i)[2] - homopiclast.at<Vec3b>(j, i)[2]) / 2;
			}
		}
		std::cout << "miss = " << miss << std::endl;
		waitKey(0);

		//////////////////////////////////////
		/////////////////合成/////////////////
		//////////////////////////////////////
		
		Mat homo2a = homo31 * homo23;
		Mat homo3a = homo21 * homo32;
		Mat homo2(homo21.size(), homo21.type()), homo3(homo21.size(), homo21.type());

		std::cout << "homo21=" << homo21 << std::endl << std::endl;
		std::cout << "homo23=" << homo23 << std::endl << std::endl;
		std::cout << "homo31=" << homo31 << std::endl << std::endl;
		std::cout << "homo32=" << homo32 << std::endl << std::endl;


		std::cout << "look!!" << std::endl;
		std::cout << "homo21=" << homo21 << std::endl << std::endl;
		std::cout << "homo2a=" << homo2a << std::endl << std::endl;

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				homo2.at<double>(j, i) = (homo2a.at<double>(j, i) + homo21.at<double>(j, i)) /2;
				homo3.at<double>(j, i) = (homo3a.at<double>(j, i) + homo31.at<double>(j, i)) /2;
			}
		}

		std::cout << "avehomo2=" << homo2 << std::endl << std::endl;


		Mat homopic2a, homopic3a;

		//AddPic をホモグラフィ行列を用いて変形し、 homopic に格納する。
		cv::warpPerspective(src2[1], homopic2a, homo2, Size(static_cast<int>(src2[1].cols), static_cast<int>(src2[1].rows)));
		imwrite(HomeDir + "/avehomo2.jpg", homopic2a);
		cv::warpPerspective(src2[2], homopic3a, homo3, Size(static_cast<int>(src2[2].cols), static_cast<int>(src2[2].rows)));
		imwrite(HomeDir + "/avehomo3.jpg", homopic3a);

		result = src2[0];
		mix2Pic(result, homopic2a, 1);
		mix2Pic(result, homopic3a, 1);

		imwrite(HomeDir + "/averesult.jpg", result);	
	}


	waitKey(0);
}

int _tmain(int argc, _TCHAR* argv[])
{
	makePanorama1();
	return 0;
}

