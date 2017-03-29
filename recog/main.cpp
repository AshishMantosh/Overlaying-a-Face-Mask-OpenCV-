#include<iostream>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<string>
#include<stdio.h>

using namespace std;

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face/facerec.hpp"

using namespace cv;
using namespace cv::face;

string face_haarcascade = "/home/amb/code/graphics/opencv/recog/haarcascade_frontalface_alt.xml";

void detectFaces(Mat frame)
{
	Mat mask = imread("/home/amb/code/graphics/opencv/recog/mask.jpg");

	CascadeClassifier f_haarcascade;
	if(!f_haarcascade.load(face_haarcascade))
	{
		cout<<"Couldn't load the haarcascade file for detecting faces."<<endl;
	}

	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);

	equalizeHist(frame_gray, frame_gray);

	f_haarcascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));

	for(int i=0;i<faces.size();i++)
	{
		Point pt1,pt2;

                pt1.x = faces[i].x;
                pt1.y = faces[i].y;

                pt2.x = faces[i].x + faces[i].width;
                pt2.y = faces[i].y + faces[i].height;

//                rectangle(frame, pt1 , pt2, Scalar(255,255,255),2,8,0);

		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);

		Mat mask1, src1;

		Size face_size = Size(faces[i].width, faces[i].height);

		resize(mask, mask1, face_size);

		Rect roi(center.x - face_size.width/2, center.y - face_size.width/2, face_size.width, face_size.width);

		frame(roi).copyTo(src1);

		Mat mask2, m, m1;

		cvtColor(mask1, mask2, CV_BGR2GRAY);
		
		threshold(mask2, mask2, 230,255,CV_THRESH_BINARY_INV);

		vector<Mat> maskChannels(3), result_mask(3);
		split(mask1, maskChannels);

		bitwise_and(maskChannels[0], mask2, result_mask[0]);
		bitwise_and(maskChannels[1], mask2, result_mask[1]);
		bitwise_and(maskChannels[2], mask2, result_mask[2]);

		merge(result_mask, m);

		mask2 = 255 -mask2;
		
		vector<Mat> srcChannels(3);
		split(src1, srcChannels);

		bitwise_and(srcChannels[0], mask2, result_mask[0]);
		bitwise_and(srcChannels[1], mask2, result_mask[1]);
		bitwise_and(srcChannels[2], mask2, result_mask[2]);
		
		merge(result_mask, m1);

		addWeighted(m,1,m1,1,0,m1);

		m1.copyTo(frame(roi));
		
		
		imwrite("picture.jpg", frame);		

	}

	return;
}	

int main(int argc, char** argv)
{
	int device_id = 0;

	VideoCapture cap(device_id);

	if(!cap.isOpened())
	{
		cout<<"Webcam didn't open."<<endl;
		exit(1);
	}

	cout<<"Press Escape Key to take a photo and terminate the webcam."<<endl;

	Mat frame;
	
	for(;;)
	{
		cap>>frame;

		detectFaces(frame);

		imshow("Face detector", frame);

		char key = (char)waitKey(10);
	
		if(key == 27)
		{
			break;
		}
	}

	return 0;
}
	
			
		
			
	
