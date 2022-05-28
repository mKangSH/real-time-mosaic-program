#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define WEBCAM 0

Mat videoFrame;

const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const String config = "deploy.prototxt";

const String clModel = "simple_frozen_graph.pb";
const String clConfig = "simple_frozen_graph.pbtxt";

void bgrHistMonitor();
vector<Rect> harrFaceClassification(Mat& src, CascadeClassifier& face_classifier);
vector<Rect> dnnFaceDetection(Mat& src, Net& net);
int dnnFaceClassification(Mat src, Net& net);
void mosaic(Mat& src, vector<Rect>& faces);
void mosaicProcess(Mat& image, vector<Rect>& face);

Net net = readNetFromCaffe(config, model);
Net net2 = readNetFromTensorflow(clModel);

void imgSave(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat test;

		test = imread("D:/test1.jpg", IMREAD_COLOR);

		dnnFaceDetection(test, net);

		imshow("dnntest", test);
	}

	else if (event == EVENT_RBUTTONDOWN) {}
	else if (event == EVENT_MBUTTONDOWN) {}

}

int main(int argc, char** argv)
{
	int videoWidth, videoHeight;
	float videoFPS;
	vector<Rect> dnn;
	int result = 0;

	VideoWriter videoWriter;
	VideoCapture videoCapture(WEBCAM);

	videoFPS = videoCapture.get(CAP_PROP_FPS);
	videoWidth = videoCapture.get(CAP_PROP_FRAME_WIDTH);
	videoHeight = videoCapture.get(CAP_PROP_FRAME_HEIGHT);

	/*
	videoWriter.open("saveData.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFPS, Size(videoWidth, videoHeight), true);

	if (!videoWriter.isOpened())
	{
		cout << "Can't write video" << endl;
		return -1;
	}
	*/

	if (!videoCapture.isOpened())
	{
		cout << "동영상 재생 불가" << endl;
		return -1;
	}

	namedWindow("dnn", 1);

	while (1)
	{
		videoCapture >> videoFrame;

		if (videoFrame.empty())
		{
			cout << "Video end" << endl;
			break;
		}

		dnn = dnnFaceDetection(videoFrame, net);

		mosaicProcess(videoFrame, dnn);

		imshow("dnn", videoFrame);

		// videoWriter << videoFrame;

		if (waitKey(1) == 27) break;
	}

	destroyWindow("dnn");

	return 0;
}

void bgrHistMonitor()
{
	Mat bHist, gHist, rHist;
	bool uniform = true;
	int histSize = 256;

	float range[] = { 0, 256 };
	const float* histRange = { range };
	vector<Mat> bgrPlanes;

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

vector<Rect> dnnFaceDetection(Mat& src, Net& net)
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

	return faces;
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

void mosaicProcess(Mat& image, vector<Rect>& faces)
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
	for (int faceCount = 0; faceCount < faces.size(); faceCount++)
	{
		if (dnnFaceClassification(image(Range(faces[faceCount].y, faces[faceCount].y + faces[faceCount].height),
			Range(faces[faceCount].x, faces[faceCount].x + faces[faceCount].width)), net2) == 2)
		{
			continue;
		}

		for (int i = 0; i < faces[faceCount].height / mb; i++) {
			for (int j = 0; j < faces[faceCount].width / mb; j++) {
				cnts = 0;
				B = 0;
				G = 0;
				R = 0;
				xStartPoint = faces[faceCount].x + (j * mb);
				yStartPoint = faces[faceCount].y + (i * mb);

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

int dnnFaceClassification(Mat src, Net& net)
{
	Mat blob = blobFromImage(src, 1 / 255.f, Size(300, 300));
	net.setInput(blob);
	Mat res = net.forward();

	cout << res << endl;

	if (res.at<float>(0, 0) > 0.5)
	{
		cout << "jmk" << endl;
		cout << res.at<float>(0, 0) << endl;
		return 1;
	}

	else
	{
		cout << "ksh" << endl;
		cout << res.at<float>(0, 0) << endl;
		return 2;
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