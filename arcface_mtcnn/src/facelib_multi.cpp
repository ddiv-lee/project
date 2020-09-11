#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
#include <stack>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "prelu.h"
#include "network.h"
#include "mtcnn.h"
//并行  
#include <thread>  
//互斥访问  
#include <mutex>  
int cutnum=0;
bool captureOpen = false;  
//读取的每张图像  
 
cv::VideoCapture capture;  
std::string url = "rtsp://admin:admin123@192.168.1.188:554/ch01.264";
//加锁器  
std::mutex mtx;  
//是否读图成功  
bool imgready = false;  
 

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1 
// stuff we know about the network and the input/output blobs

static const int RECINPUT_H = 112;
static const int RECINPUT_W = 112;
static const int RECOUTPUT_SIZE = 512;
static const int INPUT_H=1080;
static const int INPUT_W=1920;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;
static Logger gLogger;
REGISTER_TENSORRT_PLUGIN(PReluPluginCreator);
struct facemap {
	cv::Mat face;
};
struct faceres
{
    cv::Mat facemap;
    bool exist;
};


void getFiles( std::string path, std::vector<std::string>& files )  
{  

    //文件信息  
	ifstream file;
	file.open(path, ios::in);

    std::string strLine;

    while (getline(file, strLine))
	{
		if (strLine.empty())
			continue;
        files.push_back(strLine);  
	}
 
}




void warp_affine(cv::Mat &img,struct Bbox face, double scale=1.0){
    
    cv::Point2f eye_center=(cv::Point(face.ppoint[2],face.ppoint[7]));
    float dy=face.ppoint[6]-face.ppoint[5];
    float dx=face.ppoint[1]-face.ppoint[0];
    double angle=cv::fastAtan2(dy,dx);
    cv::Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat=cv::getRotationMatrix2D(eye_center,angle,scale=scale);
    cv::warpAffine(img,img, warp_mat, img.size());
    //cv::imwrite("warpface.jpg",warpface);

}




void doRECInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * RECINPUT_H * RECINPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * RECOUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * RECINPUT_H * RECINPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * RECOUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

cv::Mat resizeKeepAspectRatio(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor)
{
    cv::Mat output;

    double h1 = dstSize.width * (input.rows/(double)input.cols);
    double w2 = dstSize.height * (input.cols/(double)input.rows);
    if( h1 <= dstSize.height) {
        cv::resize( input, output, cv::Size(dstSize.width, h1));
    } else {
        cv::resize( input, output, cv::Size(w2, dstSize.height));
    }

    int top = (dstSize.height-output.rows) / 2;
    int down = (dstSize.height-output.rows+1) / 2;
    int left = (dstSize.width - output.cols) / 2;
    int right = (dstSize.width - output.cols+1) / 2;

    cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor );

    return output;
}
struct Bbox deteceface(cv::Mat &img,mtcnn find){
    cv::resize(img,img,cv::Size(INPUT_W,INPUT_H));
    cv::Mat cutfacemat;
    img.copyTo(cutfacemat);
    // prepare input data ---------------------------
    vector<struct Bbox> res;
    
    struct Bbox maxres;
    maxres.exist=0;
    maxres.y1=0;
    maxres.x1=0;
    maxres.score=0;
    // Run inference
    //std::cout<<"in deteceface"<<std::endl;
    res.clear();

    auto start = std::chrono::system_clock::now();

    find.cutFace(cutfacemat,res);
    //cout<<"out cutface"<<std::endl;
    auto end = std::chrono::system_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    cv::Rect full_img(0,0,img.cols,img.rows);
    for(vector<struct Bbox>::iterator it=res.begin(); it!=res.end();it++){
    if((*it).exist){
        //cout<<"face score:"<<(*it).score<<std::endl;
        //cout<<"find face"<<std::endl;
        if ((*it).score < 0.1) continue;
            if (maxres.score<(*it).score){
        	        maxres=*it; }
        //rectangle(img, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0,0,255), 2,8,0);
        //for(int num=0;num<5;num++)circle(img,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);

    
    }
    //cv::imwrite("detect.jpg",img);
    }
    
    //cv::Rect box = get_rect_adapt_landmark(img, maxres.bbox, maxres.landmark);
    
    //cout<<maxres.y1<<","<<maxres.x1<<std::endl;
    //cout<<maxres.y2<<","<<maxres.x2<<std::endl;
    
    //cout<<maxres.exist<<std::endl;
    //std::cout<<maxres.score<<std::endl;
    
    
    return maxres;

}
/*
cv::Mat detecefacetest(cv::Mat &img,IExecutionContext* context){
   
    // prepare input data ---------------------------
    static float data[3 * INPUT_H * INPUT_W];
    cv::Mat pr_img = preprocess_img(img);
    //cv::imwrite("preprocessed.jpg", pr_img);
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
        data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
        data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
    }
    // Run inference
    static float prob[OUTPUT_SIZE];
    std::vector<decodeplugin::Detection> res;
   
    res.clear();
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, 1);
    nms(res, prob);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    cv::Rect full_img(0,0,img.cols,img.rows);
    std::cout << "detected before nms -> " << prob[0] << std::endl;
    std::cout << "after nms -> " << res.size() << std::endl;
    cv::Mat FACE;
    if(res.size()>0){
    cv::Rect maxface;
    int maxface_ind;
    for (size_t j = 0; j < res.size(); j++) {
        if (res[j].class_confidence < 0.1) continue;
        
        cv::Rect r = get_rect_adapt_landmark(img, res[j].bbox, res[j].landmark);
        if (maxface.area()<r.area()){
        	maxface=r;
                maxface_ind=j;
                 
        }
    }
    FACE = img(maxface&full_img);
    
    FACE=resizeKeepAspectRatio(FACE,cv::Size(112,112),cv::Scalar(0,0,0));


     
    cv::rectangle(img, maxface, cv::Scalar(0x27, 0xC1, 0x36), 2);
      for (int k = 0; k < 10; k += 2) {
            cv::circle(img, cv::Point(res[maxface_ind].landmark[k], res[maxface_ind].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
        }
    //cv::imwrite("result.jpg", img);
    return FACE;}
    else{
        return FACE;
    }

}*/
cv::Mat feature_lib(cv::Mat &img,IExecutionContext* context){

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * RECINPUT_H * RECINPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * RECOUTPUT_SIZE];
        for (int i = 0; i < RECINPUT_H * RECINPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        data[i + RECINPUT_H * RECINPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * RECINPUT_H * RECINPUT_W] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
    }
        // Run inference
    auto start = std::chrono::system_clock::now();
    doRECInference(*context, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out(1,512, CV_32FC1, prob);
    cv::Mat out_norm;
    cv::normalize(out, out_norm);
    return out_norm;
}
cv::Mat recognition(cv::Mat &img,IExecutionContext* context){

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * RECINPUT_H * RECINPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * RECOUTPUT_SIZE];
        for (int i = 0; i < RECINPUT_H * RECINPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        data[i + RECINPUT_H * RECINPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * RECINPUT_H * RECINPUT_W] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
    }
        // Run inference
    auto start = std::chrono::system_clock::now();
    doRECInference(*context, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out(512, 1, CV_32FC1, prob);
    cv::Mat out_norm;
    cv::normalize(out, out_norm);
    return out_norm;
}
struct faceres detect_wrap(cv::Mat &img,mtcnn find){
    cv::Mat facemat;
    img.copyTo(facemat);
    struct Bbox maxres=deteceface(facemat,find);
    struct faceres FACEwraped;
    FACEwraped.exist=0;
    cv::Mat img_copy;
    img.copyTo(img_copy);
      //cv::imshow("facedetect",img);
    //    cv::waitKey(0);

    if (maxres.exist){
        rectangle(img, Point((maxres).y1, (maxres).x1), Point((maxres).y2, (maxres).x2), Scalar(0,0,255), 2,8,0);
        for(int num=0;num<5;num++)circle(img,Point((int)(maxres.ppoint[num]), (int)(maxres.ppoint[num+5])),3,Scalar(0,255,255), -1);
 
        cv::Rect full_img(0,0,INPUT_W,INPUT_H);   
        cv::Rect r=cv::Rect(maxres.y1,maxres.x1,(maxres.y2-maxres.y1),(maxres.x2-maxres.x1));
       // cout<<"rect1:"<<r<<std::endl;
        cv::Rect maxfacewrap=cv::Rect(r.x, r.y, r.width,r.height);
       // cout<<"rect2:"<<maxfacewrap<<std::endl;
        FACEwraped.facemap = img_copy(maxfacewrap&full_img);
        FACEwraped.exist=1;
        //std::cout<<(maxfacewrap&full_img)<<std::endl;
        //cv::imshow("facedetext",img);
        //cv::waitKey(0);
        //cv::imshow("facewrape",FACEwraped.facemap);
        //cv::waitKey(0);
        warp_affine(img_copy,maxres, 1.0);
        img_copy.copyTo(facemat);
        struct Bbox wrapedres=deteceface(facemat,find);
        //std::cout<<"wrapeders="<<wrapedres.exist<<std::endl;

        if (wrapedres.exist){
        //rectangle(img_copy, Point((wrapedres).y1, (wrapedres).x1), Point((wrapedres).y2, (wrapedres).x2), Scalar(0,0,255), 2,8,0);
        //for(int num=0;num<5;num++)circle(img_copy,Point((int)(wrapedres.ppoint[num]), (int)(wrapedres.ppoint[num+5])),3,Scalar(0,255,255), -1);
 
       // imshow("img_wraped",img_copy);
       // cv::waitKey(0);
        cv::Rect r(wrapedres.y1,wrapedres.x1,(wrapedres.y2-wrapedres.y1),(wrapedres.x2-wrapedres.x1));
        //cout<<"rect3:"<<r<<std::endl;
        cv::Rect wraped_img(0,0,img_copy.cols,img_copy.rows);
        FACEwraped.facemap = img_copy(r&wraped_img);
        //imshow("resizeface",FACEwraped.facemap);
        //cv::waitKey(0);

    }}
    if (FACEwraped.exist){
        FACEwraped.facemap=resizeKeepAspectRatio(FACEwraped.facemap,cv::Size(112,112),cv::Scalar(0,0,0));
        //imshow("112,112",FACEwraped.facemap);
        //cv::waitKey(0);
        std::string storename="./result/"+to_string(cutnum)+".jpg";
        cout<<"store image:"<<storename<<std::endl;
        cv::imwrite(storename, FACEwraped.facemap);
        cutnum+=1;
    }

    
    return FACEwraped;
}
std::map<std::string,facemap> facelib(std::string path_root,IExecutionContext* reccontext,mtcnn find){
    std::vector<std::string> files;  
    getFiles(path_root, files);
    std::map<std::string, facemap> faceid;
    for (int i = 0;i < files.size();i++)  
{  
	//std::cout<<files[i].c_str()<<std::endl;
	std::string imgpath = "/home/lee/share/facedemo/yuangongxinxi/" + files[i];
    std::cout<<imgpath<<std::endl;
	cv::Mat img = cv::imread(imgpath);
    //cv::imshow("face",img);
    //cv::waitKey(0);
    img=resizeKeepAspectRatio(img,cv::Size(INPUT_W,INPUT_H),cv::Scalar(0,0,0));
    //cv::resize(img,img,cv::Size(INPUT_W,INPUT_H));

    
    struct faceres FACEwraped;
    FACEwraped=detect_wrap(img,find);
    if (FACEwraped.exist){
    cv::Mat feature_map=recognition(FACEwraped.facemap,reccontext);
    
	facemap faceroi = { feature_map };
	faceid.insert(std::pair<std::string, facemap>(files[i], faceroi));
    }
    else{
        continue;
    }

}
    return faceid;
}

/** 
* @brief 读图 
* 
* @return Mat 
*/  
void captureThread(stack<cv::Mat> &frame_q)  
{  
cv::Mat frame;

//打开图像  
capture.open(url);  


while (1)  
{  
//cout<<"111111"<<std::endl;
//加锁  
mtx.lock();
if(frame_q.size()>50){
    while (!frame_q.empty()){
        frame_q.pop();

    }}
    capture>>frame;
//cout<<"thread1"<<std::endl;
frame_q.push(frame); 
//cout<<"push"<<std::endl;
//解锁  
mtx.unlock();  
//cout<<"end"<<std::endl;
}
}

string getTime()
{
    time_t timep;
    time (&timep); //获取time_t类型的当前时间
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S",localtime(&timep) );//对日期和时间进行格式化
    return tmp;

}
int main(int argc, char** argv) {
    //arg 
    stack<cv::Mat> frame_q;
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        //std::cerr << "./retina_r50 -s   // serialize model to plan file" << std::endl;
        std::cerr << "./retina_r50  // list path" << std::endl;
        return -1;
    }

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream

    char *recModelStream{nullptr};
    size_t size{0};

//-------------------------------------------------------------------------

//face recognition engine deserialize--------------------------------------
        std::ifstream recfile("arcface-r50.engine", std::ios::binary);
        if (recfile.good()) {
            recfile.seekg(0, recfile.end);
            size = recfile.tellg();
            recfile.seekg(0, recfile.beg);
            recModelStream = new char[size];
            assert(recModelStream);
            recfile.read(recModelStream, size);
            recfile.close();
        }

    std::string filepath=argv[1];

//recognition engine--------------------------------------------------------------------------------------
    IRuntime* recruntime = createInferRuntime(gLogger);
    assert(recruntime != nullptr);
    ICudaEngine* recengine = recruntime->deserializeCudaEngine(recModelStream, size);
    assert(recengine != nullptr);
    IExecutionContext* reccontext = recengine->createExecutionContext();
    assert(reccontext != nullptr);
//-----------------------------------------------------------------------------------------------------
/*debug
std::map<std::string, facemap> facemaplib;
for(;;){
    thread t1(captureThread);  
    t1.join(); 
    //cv::Mat face2=detecefacetest(frame,detectcontext);
    //decodeplugin::Detection maxres=deteceface(frame,detectcontext);
    cv::Mat face2=detect_wrap(frame,detectcontext);
    imshow("Live", frame);
    cv::waitKey(1);
}
*/

    mtcnn find(INPUT_H, INPUT_W);
    
    std::map<std::string, facemap> facemaplib;
    facemaplib=facelib(filepath,reccontext,find);
    thread t1(captureThread,ref(frame_q));  
    t1.detach(); 
    cv::Mat frame;
    for(;;){ 


    mtx.lock();
    if(!frame_q.empty()){
    //cout<<"copy"<<std::endl;
    frame_q.top().copyTo(frame);
    frame_q.pop();}
    mtx.unlock();
    if (frame.rows>0 && frame.cols>0){
    cv::Mat feature_map2;
   
    struct faceres face2=detect_wrap(frame,find);
    //imshow("Live", frame);
    //cv::waitKey(1);
    if(face2.exist){
    //std::cout<<"in face2="<<std::endl;
    
    feature_map2=recognition(face2.facemap,reccontext);
    std::map<std::string, facemap>::iterator iter;
    std::map<std::string,float> resultmap;
    std::string samefaceid;
    float sameface_score=100;
        for (iter = facemaplib.begin(); iter != facemaplib.end(); iter++) {

            //std::cout << iter->first<< std::endl;
            std::string imgname=iter->first;
            cv::Mat img = (iter->second).face;
            //cv::Mat result = img * feature_map2;
            cv::Mat result = img - feature_map2;
            result=result.mul(result);
            cv::Scalar s=sum(result);
            //cout<<"s:"<<s;
            //resultmap.insert(std::pair<std::string, float>(imgname, *(float*)result.data));
            resultmap.insert(std::pair<std::string, float>(imgname, (float)s[0]));
            //std::cout << "similarity scor: " << *(float*)result.data << std::endl;
            if (sameface_score>(float)s[0]){
            samefaceid=imgname;
            sameface_score=(float)s[0];
        }
    }
    cout<<"name:"<<samefaceid;
    cout<<"score"<<sameface_score;
    string   time = getTime();
    cout << time << std::endl;

    if(sameface_score<1.0){

    ofstream dout;
    std::string jpgname="./result/"+to_string(cutnum-1)+".jpg";
    dout.open("result.txt",ios::app);
    dout<<jpgname<<" "<<samefaceid<<" "<<sameface_score<<" "<<time<<std::endl;
    dout.close();
	}
        }
    }
        //cout<<"out face2"<<std::endl;
    }

    

 // Destroy the engine
  
  

    reccontext->destroy();
    recengine->destroy();
    recruntime->destroy();

    return 0;
}
    
