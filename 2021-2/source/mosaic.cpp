#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define WEBCAM 0

int mx1, my1, mx2, my2;

Mat videoFrame;
VideoCapture videoCapture(WEBCAM);
VideoWriter videoWriter, videoWriter1, videoWriter2;
vector<Mat> bgrPlanes;
float videoFPS;
int videoWidth, videoHeight;
int histSize = 256;
float range[] = { 0, 256 };
const float* histRange = { range };
bool uniform = true, accumulate = false;
Mat bHist, gHist, rHist, equalizedFrame, grayFrame;

const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const String config = "deploy.prototxt";

void bgrHistMonitor();
vector<Rect> harrFaceClassification(Mat& src, CascadeClassifier& face_classifier);
void dnnFaceClassification(Mat& src, Net& net);
void mosaic(Mat& src, vector<Rect>& faces);
void mosaicProcess(Mat& image, vector<Rect>& face);

CascadeClassifier face_classifier; // 비올라-존스 알고리즘 클래스
CascadeClassifier front_face_classifier;
Net net = readNetFromCaffe(config, model);

void imgSave(int event, int x, int y, int flags, void* param)
{
	vector<Rect> profile;
	vector<Rect> frontFace;
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat test, test1, test2;

		test = imread("D:/test1.jpg", IMREAD_COLOR);
		test1 = test.clone();
		test2 = test.clone();

		dnnFaceClassification(test, net);
		profile = harrFaceClassification(test1, face_classifier);
		frontFace = harrFaceClassification(test2, front_face_classifier);

		mosaicProcess(test1, profile);
		mosaicProcess(test2, frontFace);

		imshow("dnntest", test);
		imshow("harr side test", test1);
		imshow("harr front test", test2);
	}
	
	else if (event == EVENT_RBUTTONDOWN) {}
	else if (event == EVENT_MBUTTONDOWN) {}
	
}

int main(int argc, char** argv)
{
	vector<Rect> profile;
	vector<Rect> frontFace, dnn;
	Mat haarVideoFrameSide, haarVideoFrameFront;
	
	videoFPS = videoCapture.get(CAP_PROP_FPS);
	videoWidth = videoCapture.get(CAP_PROP_FRAME_WIDTH);
	videoHeight = videoCapture.get(CAP_PROP_FRAME_HEIGHT);

	videoWriter.open("saveData.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFPS, Size(videoWidth, videoHeight), true);
	videoWriter1.open("saveData1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFPS, Size(videoWidth, videoHeight), true);
	videoWriter2.open("saveData2.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFPS, Size(videoWidth, videoHeight), true);

	if (!videoWriter.isOpened())
	{
		cout << "Can't write video" << endl;
		return -1;
	}

	face_classifier.load("D:/openCV-4.5.3/etc/haarcascades/haarcascade_profileface.xml");
	front_face_classifier.load("D:/openCV-4.5.3/etc/haarcascades/haarcascade_frontalface_default.xml");

	if (face_classifier.empty() || front_face_classifier.empty())
	{
		cout << "xml load error" << endl;
		return -1;
	}

	if (!videoCapture.isOpened())
	{
		cout << "동영상 재생 불가" << endl;
		return -1;
	}

	namedWindow("dnn", 1);
	setMouseCallback("dnn", imgSave, &videoFrame);

	while (1)
	{
		videoCapture >> videoFrame;
		haarVideoFrameFront = videoFrame.clone();
		haarVideoFrameSide = videoFrame.clone();

		if (videoFrame.empty())
		{
			cout << "Video end" << endl;
			break;
		}

		dnnFaceClassification(videoFrame, net);
		profile = harrFaceClassification(haarVideoFrameSide, face_classifier);
		frontFace = harrFaceClassification(haarVideoFrameFront, front_face_classifier);
		
		mosaicProcess(haarVideoFrameSide, profile);
		mosaicProcess(haarVideoFrameFront, frontFace);

		imshow("dnn", videoFrame);
		imshow("haar filter", haarVideoFrameFront);
		imshow("haar filter2", haarVideoFrameSide);

		videoWriter << videoFrame;
		videoWriter1 << haarVideoFrameFront;
		videoWriter2 << haarVideoFrameSide;

		if (waitKey(1) == 27) break;
	}

	destroyWindow("dnn");
	destroyWindow("haar filter");

	return 0;
}

void bgrHistMonitor()
{
	split(videoFrame, bgrPlanes);
	calcHist(&bgrPlanes[0], 1, 0, Mat(), bHist, 1, &histSize, &histRange, uniform, false);
	calcHist(&bgrPlanes[1], 1, 0, Mat(), gHist, 1, &histSize, &histRange, uniform, false);
	calcHist(&bgrPlanes[2], 1, 0, Mat(), rHist, 1, &histSize, &histRange, uniform, false);

	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)(hist_w / histSize));

	Mat histImageB(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageG(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageR(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	normalize(bHist, bHist, 0, histImageB.rows, NORM_MINMAX, -1, Mat());
	normalize(gHist, gHist, 0, histImageG.rows, NORM_MINMAX, -1, Mat());
	normalize(rHist, rHist, 0, histImageR.rows, NORM_MINMAX, -1, Mat());

	for (int i = 0; i < 255; i++)
	{
		line(histImageB, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - bHist.at<float>(i)), Scalar(255, 0, 0));
		line(histImageG, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - gHist.at<float>(i)), Scalar(0, 255, 0));
		line(histImageR, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - rHist.at<float>(i)), Scalar(0, 0, 255));
	}

	histImageB.push_back(histImageG);
	histImageB.push_back(histImageR);

	imshow("히스토그램 모니터", histImageB);
}

vector<Rect> harrFaceClassification(Mat& src, CascadeClassifier& face_classifier)
{
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	equalizeHist(src_gray, src_gray);

	vector<Rect> faces;

	face_classifier.detectMultiScale(src_gray, faces,

		1.1, // increase search scale by 10% each pass

		3,   // merge groups of three detections

		CASCADE_FIND_BIGGEST_OBJECT | CASCADE_SCALE_IMAGE,

		Size(30, 30)
	);

	for (int i = 0; i < faces.size(); i++)
	{
		Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

		Point tr(faces[i].x, faces[i].y);

		rectangle(src, lb, tr, Scalar(0, 255, 0), 3, 4, 0);
	}

	return faces;
}

void dnnFaceClassification(Mat& src, Net& net)
{
	Mat blob = blobFromImage(src, 1, Size(300, 300), Scalar(104, 177, 123));
	net.setInput(blob);
	Mat res = net.forward();
	Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>());
	Mat src_mosaic;
	Mat img_temp;
	vector<Rect> faces;

	for (int i = 0; i < detect.rows; i++) {
		float confidence = detect.at<float>(i, 2);
		if (confidence < 0.5)
			break;

		int x1 = cvRound(detect.at<float>(i, 3) * src.cols);
		int y1 = cvRound(detect.at<float>(i, 4) * src.rows);
		int x2 = cvRound(detect.at<float>(i, 5) * src.cols);
		int y2 = cvRound(detect.at<float>(i, 6) * src.rows);

		rectangle(src, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 255, 0), 3, 4, 0);
		faces.push_back(Rect(Point(x1, y1), Point(x2, y2)));
		//String label = format("Face: %4.3f", confidence);
		//putText(src, label, Point(x1, y1 - 1), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0));
	}
	mosaicProcess(src, faces);
}

void mosaic(Mat& src, vector<Rect>& faces)
{
	Mat src_mosaic;

	for (int i = 0; i < faces.size(); i++) 
	{
		Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

		Point tr(faces[i].x, faces[i].y);

		rectangle(src, lb, tr, Scalar(0, 255, 0), 3, 4, 0);

		src_mosaic = src(Rect(lb, tr));
		Mat img_temp;

		resize(src_mosaic, img_temp, Size(src_mosaic.rows / 16, src_mosaic.cols / 16));
		resize(img_temp, src_mosaic, Size(src_mosaic.rows, src_mosaic.cols));
	}
}

void mosaicProcess(Mat& image, vector<Rect>& face)
{
	int cnts = 0;
	int mb = 9;
	int wPoint = 0;
	int hPoint = 0;
	int xStartPoint = 0;
	int yStartPoint = 0;
	double R = 0;
	double G = 0;
	double B = 0;
	for (int faceCount = 0; faceCount < face.size(); faceCount++)
	{
		for (int i = 0; i < face[faceCount].height / mb; i++) {
			for (int j = 0; j < face[faceCount].width / mb; j++) {
				cnts = 0;
				B = 0;
				G = 0;
				R = 0;
				xStartPoint = face[faceCount].x + (j * mb);
				yStartPoint = face[faceCount].y + (i * mb);

				// 이미지의 픽셀 값의 r, g, b 값의 각각 합을 구함
				for (int mbY = yStartPoint; mbY < yStartPoint + mb; mbY++) {
					for (int mbX = xStartPoint; mbX < xStartPoint + mb; mbX++) {
						wPoint = mbX;
						hPoint = mbY;

						if (mbX >= image.cols) {
							wPoint = image.cols - 1;
						}
						if (mbY >= image.rows) {
							hPoint = image.rows - 1;
						}

						Vec3b color = image.at<Vec3b>(hPoint, wPoint);
						B += color.val[0];
						G += color.val[1];
						R += color.val[2];
						cnts++;
					}
				}

				// r, g, b 값의 평균 산출
				B /= cnts;
				G /= cnts;
				R /= cnts;

				// 모자이크 색상 생성
				Scalar color;
				color.val[0] = B;
				color.val[1] = G;
				color.val[2] = R;

				// 프레임에 모자이크 이미지 삽입
				rectangle(
					image,
					Point(xStartPoint, yStartPoint),
					Point(xStartPoint + mb, yStartPoint + mb),
					color,
					FILLED,
					8,
					0
				);
			}
		}
	}
}

/*
	videoFPS = videoCapture.get(CAP_PROP_FPS);
	videoWidth = videoCapture.get(CAP_PROP_FRAME_WIDTH);
	videoHeight = videoCapture.get(CAP_PROP_FRAME_HEIGHT);
	videoWriter.open("test.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFPS, Size(videoWidth, videoHeight), true);
	if (!videoWriter.isOpened())
	{
		cout << "Can't write video" << endl;
		return -1;
	}
		//videoWriter << videoFrame;
	*/