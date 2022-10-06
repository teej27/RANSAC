#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>


int main()
{
	// loading images
	cv::Mat img1 = cv::imread ("imgL.png");
	cv::Mat img2 = cv::imread ("imgR.png");
	if ( !img1.data || !img2.data )
	{
		std::cout << "Reading picture error！" << std::endl;
		exit (1);
	}

	// create SIFT class pointer
	cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create ();
	// calculate run time
	double timeBegin = cv::getTickCount ();
	
	// detect the keypoints
	// keypoint :pt(x,y):coordinate , size():diameter of neighbourhood , angle():direction of point
	// response:indicator of how good,the higher,the better, octave: pyramid octave in which the keypoint has been detected , class_id: object id defaulted -1
	std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
	cv::Mat descriptors1, descriptors2;

	f2d->detect(img1, keyPoints1);
	f2d->detect(img2, keyPoints2);
	std::cout << "The keypoints number of img1:" << keyPoints1.size() << std::endl;
	std::cout << "The keypoints number of img2:" << keyPoints2.size() << std::endl;

	// calculate descriptors (feature vectors)
	f2d->compute(img1, keyPoints1, descriptors1);
	f2d->compute(img2, keyPoints2, descriptors2);

	// coutting time frequency unit
	double freq = cv::getTickFrequency();
	double tt = ((double)cv::getTickCount() - timeBegin) / freq;
	std::cout << "Extract SIFT Time:" << tt << "ms" << std::endl;

    //
	cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(2, false);
	
    //
	std::vector<cv::DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches);
	std::cout << "The number of match:" << matches.size() << std::endl;
	
	// draw the match result
	cv::Mat imgMatches;
	cv::drawMatches(img1, keyPoints1, img2, keyPoints2, matches, imgMatches, cv::Scalar::all(-1));
	cv::namedWindow ("SIFT with no RANSAC", 0);
	cv::imshow ("SIFT with no RANSAC", imgMatches);


	std::vector<cv::DMatch> matches2 = matches;

	size_t numPt = matches2.size();

	if ( numPt < 100 )
	{
		std::cout << "Don't find enough match point pairs" << std::endl;
		exit(1);
	}

	std::vector <cv::KeyPoint> corPt1, corPt2;
	for (size_t i = 0; i < numPt; i++)
	{
		corPt1.push_back( keyPoints1.at( matches2.at(i).queryIdx ) );
		corPt2.push_back( keyPoints2.at( matches2.at(i).trainIdx ) );

	}

	// change the type from Keypoint to Point2f for using findFundamentalMat() 
	std::vector <cv::Point2f> p1, p2;
	for (size_t i = 0; i < numPt; i++)
	{
		p1.push_back(corPt1.at(i).pt);
		p2.push_back(corPt2.at(i).pt);
	}

	// ransac flag vector, inliers will be 1 and outliers will be 0 
	std::vector<uchar> RansacStatus;
	// calculate Fundamental matrix
	cv::Mat Fundamental = findFundamentalMat (p1, p2, RansacStatus, cv::FM_RANSAC);
	// store key points after RANSAC by keyPointAfterR1 & 2
	std::vector <cv::KeyPoint> keyPointR1, keyPointR2;
	std::vector <cv::DMatch> ransacMatches;

	size_t corIndex = 0;
	for (size_t i = 0; i < numPt; i++)
	{
		if (RansacStatus.at(i) != 0)
		{
			keyPointR1.push_back( corPt1.at(i) );
			keyPointR2.push_back( corPt2.at(i) );
			matches2.at (i).queryIdx = corIndex;
			matches2.at (i).trainIdx = corIndex;
			ransacMatches.push_back( matches2.at(i) );
			corIndex++;
		}
	}
	std::cout << "after RANSAC,the match pairs number:" << ransacMatches.size() << std::endl;
	cv::Mat imgRansac;
	drawMatches(img1, keyPointR1, img2, keyPointR2, ransacMatches, imgRansac);
	cv::namedWindow ("SIFT with RANSAC", 0);
	cv::imshow("SIFT with RANSAC", imgRansac);
	cv::waitKey(0);
	return 0;
}