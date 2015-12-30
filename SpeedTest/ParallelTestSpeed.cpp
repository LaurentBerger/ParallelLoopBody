#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;


int     fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
double  fontScale = 0.5;
int     thickness = 3;
int     baseline=0;
int     bestNumberOfthread=-1;
int     maxThread=20;
int     nbMinThread=1;

class ParallelMeanLine: public cv::ParallelLoopBody
{
private:
    cv::Mat &imgSrc;
    vector<double> &meanLine;
    vector<double> &stdLine;
    bool verbose;

public:
ParallelMeanLine(Mat& img, vector<double> &m,vector<double> &s):
    imgSrc(img),
    meanLine(m),
    stdLine(s),
    verbose(false)
{}
void Verbose(bool b){verbose=b;}
virtual void operator()( const Range& range ) const 
{

    int h = imgSrc.cols;
    if (verbose)
        cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end<<" ("<<range.end-range.start<<" loops)" << endl;
// First mean and variance for each line are computed
    for (int y = range.start; y < range.end; y++)
    {

        uchar *ptr=imgSrc.ptr(y);
        int m = 0;
        int s2=0;
        for (int x = 0; x<h; x++,ptr++)
        {
                m += *ptr;
                s2 += *ptr * *ptr;
        }
        meanLine[y] = m / double(h);
        stdLine[y] = s2/h-meanLine[y]*meanLine[y];
    }
// We want : all lines must have a mean value of 128 and standard deviation of 64. Negative value or value greater than 255 are threshold
    for (int y = range.start; y < range.end; y++)
    {

        uchar *ptr=imgSrc.ptr(y);
        int v;
        for (int x = 0; x<h; x++,ptr++)
        {
               v = (*ptr-meanLine[y])*sqrt(64*64/stdLine[y])+128;
               if (v>255)
                   *ptr=255;
               else if (v<0)
                   *ptr=0;
               else 
                   *ptr=v;
        }
    }

}
};

void DrawResults(Mat &curve,vector<double> &tps,String label,double maxTime,int bestNumberOfthread)
{
#define CURVE_ROWS 800
#define CURVE_COLS 1000
    double ratioLegendx=1/5.0;
    double ratioLegendy=4/5.0;
    int    offsetRows=20;
    for (int i = 0; i < 4; i++)
    {
        String s(format("%d ms",int(i/3.0*maxTime)));
        Size textSize = getTextSize(s, fontFace,fontScale, thickness, &baseline); 
        putText(curve, s,Point(0,curve.rows*(1-i/3.0)*ratioLegendy+textSize.height/2+offsetRows),fontFace,fontScale,Scalar(255,255,255));
        line(curve,Point(CURVE_COLS*ratioLegendx, curve.rows*(1-i/3.0)*ratioLegendy+offsetRows),Point(CURVE_COLS,curve.rows*(1-i/3.0)*ratioLegendy+offsetRows),Scalar(128,128,128),1);
    }
    for (int i = 0; i < maxThread; i+= maxThread/3)
    {
            String s(format("%d threads",i));
            Size textSize = getTextSize(s, fontFace,fontScale, thickness, &baseline);           
            putText(curve, s,Point(((1-ratioLegendx)*(i)/maxThread+ratioLegendx)*CURVE_COLS-textSize.width/2,curve.rows*ratioLegendy+2*textSize.height+offsetRows),fontFace,fontScale,Scalar(255,255,255));
            //line(curve,Point(CURVE_COLS*ratioLegendx, curve.rows*(1-i/3.0)*ratioLegendy),Point(CURVE_COLS,curve.rows*(1-i/3.0)*ratioLegendy),Scalar(128,128,128),1);
    }
    // Y axis
    line(curve,Point(CURVE_COLS*ratioLegendx,offsetRows),Point(CURVE_COLS*ratioLegendx, curve.rows*ratioLegendy+offsetRows),Scalar(255,255,255),3);
    // X axis
    line(curve,Point(CURVE_COLS*ratioLegendx, curve.rows*ratioLegendy+offsetRows),Point(CURVE_COLS,curve.rows*ratioLegendy+offsetRows),Scalar(255,255,255),3);
    Point centre;
    Point pPrec(-1,-1);
    for (int nthreads = nbMinThread; nthreads < maxThread; nthreads++)
    {
        centre  = Point(((1-ratioLegendx)*nthreads/maxThread+ratioLegendx)*CURVE_COLS, curve.rows*(1-tps[nthreads]/maxTime)*ratioLegendy+offsetRows);
        circle(curve,centre,3,Scalar(0,0,255));
        if (nthreads == bestNumberOfthread)
        {
            circle(curve,centre,5,Scalar(0,255,0),2);
        }
        if (nthreads == nbMinThread)
        {
            Size textSize = getTextSize(label, fontFace,fontScale, thickness, &baseline); 
            putText(curve, label,centre-Point(textSize.width/2,-textSize.height/2),fontFace,fontScale,Scalar(255,255,255));
        }
        if (pPrec.x != -1)
        {
            line(curve,pPrec,centre,Scalar(64,64,64),1);
        }
        pPrec=centre;

    }
}

int main (int argc,char **argv)
{
    Mat m(16,16,CV_8UC1);
    
    Mat     curve = Mat::zeros(CURVE_ROWS,CURVE_COLS,CV_8UC3);
    vector<double> zoom = {1,2,4,10,16,20,30,40,50};
    int     nbTest=100;
    double  maxTime=-1;
    randn(m,128,10);
    // Look for otpimum using different image sizes
    for (int indZoom = zoom.size()-1;indZoom>=1;indZoom--)
    {
        Mat mz;
        resize(m, mz, Size(),zoom[indZoom],zoom[indZoom]);
        waitKey(1);
        vector<vector<double> > mean;
        vector<vector<double> > std;
        double  bestTime=DBL_MAX;
        vector<double> tps;
        int nthreads=nbMinThread;

        tps.resize(maxThread);
        mean.resize(maxThread);
        std.resize(maxThread);
        mean[nthreads].resize(mz.rows);
        std[nthreads].resize(mz.rows);
        Mat r;
        for ( nthreads = nbMinThread; nthreads < maxThread; nthreads++)
        {
            r = mz.clone();
            mean[nthreads].resize(r.rows);
            std[nthreads].resize(r.rows);
            ParallelMeanLine x(r,mean[nthreads],std[nthreads]);
            setNumThreads(nthreads);
            int64 tpsIni = getTickCount();
            for (int k = 0; k < nbTest; k++)
            {
                parallel_for_(cv::Range(0,r.rows-1), x,nthreads);
            }
            int64  tpsFin = getTickCount();
            tps[nthreads]=(tpsFin - tpsIni) / cvGetTickFrequency()/nbTest;
            cout << "For " << nthreads << " thread times is " << tps[nthreads] << "\n";
            cout << "*****************************************************************************************\n";
       
            if (tps[nthreads] < bestTime)
            {
                bestTime=tps[nthreads];
                bestNumberOfthread = nthreads;
            }
            if (tps[nthreads] > maxTime)
               maxTime=tps[nthreads];

        }
        DrawResults(curve,tps,format("%d",r.rows*r.cols), maxTime, bestNumberOfthread);

        bool test=true;
        for (int nthreads = nbMinThread+1; nthreads < maxThread; nthreads++)
            for (int j=0;j<m.rows;j++)
                if (mean[nthreads][j] - mean[nbMinThread][j] != 0)
                {
                    cout << "Problem with thread " << nthreads << " at row " << j << "\n";
                    test =false;
                }
        if (test)
            cout << "No problem with result\n";
        else
            cout << "Problem with result\n";
        cout << "For zoom = "<<zoom[indZoom]<<" image size is " <<r.size() << "pixels\n";
        cout << "Best configuration for this thread is for " <<bestNumberOfthread << " threads with time " << bestTime << "ms\n";
        imshow("Time function as number of thread",curve);
        waitKey(1);
    }

    waitKey();
    return 0;
}

