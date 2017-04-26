#include "main.h"
#include "CombinedSolver.h"
#include "SFSSolverInput.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#include <fstream>


int main(int argc, const char * argv[])
{
    std::string  depthInput = "../data/shape_from_shading/targetDepth.jpg";
    std::string  intensityInput = "../data/shape_from_shading/targetIntensity.jpg";

    cv::Mat      depthMap, intensityMap, initialUnknown, maskEdgeMap;

    depthMap     = imread(depthInput,CV_LOAD_IMAGE_GRAYSCALE);
    initialUnknown = Mat(depthMap.rows, depthMap.cols,depthMap.type());
    maskEdgeMap = Mat(depthMap.rows,depthMap.cols,CV_8UC1, cv::Scalar(0));
    depthMap.copyTo(initialUnknown);
    intensityMap = imread(intensityInput);
    cvtColor(intensityMap, intensityMap, CV_BGR2GRAY); // transform to 3-channels

    imshow("depthMap", depthMap);
    imshow("intensityMap", intensityMap);
    cv::waitKey(30);
    char *h_depth = new char[depthMap.rows * depthMap.cols];
    char *h_intensity = new char[depthMap.rows * depthMap.cols];
    char *h_mask = new char[depthMap.rows * depthMap.cols];
    h_depth = (char*)depthMap.data;
    h_intensity = (char*)intensityMap.data;
    h_mask = (char*)maskEdgeMap.data;

    ofstream myFile ("../data/shape_from_shading/marta_targetDepth.imagedump", ios::out | ios::binary);
    myFile.write (h_depth, depthMap.rows * depthMap.cols);
    ofstream myFile2("../data/shape_from_shading/marta_initialUnknown.imagedump", ios::out | ios::binary);
    myFile2.write (h_depth, depthMap.rows * depthMap.cols);
    ofstream myFile3("../data/shape_from_shading/marta_maskEdgeMap.imagedump", ios::out | ios::binary);
    myFile3.write (h_mask, depthMap.rows * depthMap.cols);
    ofstream myFile4("../data/shape_from_shading/marta_targetIntensity.imagedump", ios::out | ios::binary);
    myFile4.write (h_intensity, depthMap.rows * depthMap.cols);
    myFile.close();
    myFile2.close();
    myFile3.close();
    myFile4.close();

    std::string inputFilenamePrefix = "../data/shape_from_shading/marta";
    if (argc >= 2) {
        inputFilenamePrefix = std::string(argv[1]);
    }

    bool performanceRun = false;
    if (argc > 2) {
        if (std::string(argv[2]) == "perf") {
            performanceRun = true;
        }
        else {
            printf("Invalid second parameter: %s\n", argv[2]);
        }
    }

    SFSSolverInput solverInputCPU, solverInputGPU;
    solverInputGPU.load(inputFilenamePrefix, true);

    solverInputGPU.targetDepth->savePLYMesh("sfsInitDepth.ply");
    solverInputCPU.load(inputFilenamePrefix, false);

    CombinedSolverParameters params;
    params.nonLinearIter = 60;
    params.linearIter = 10;
    params.useOpt = true;
    if (performanceRun) {
        params.useCUDA  = false;
        params.useOpt   = true;
        params.useOptLM = true;
        params.useCeres = true;
        params.nonLinearIter = 60;
        params.linearIter = 10;
    }

    CombinedSolver solver(solverInputGPU, params);
    printf("Solving\n");
    solver.solveAll();
    std::shared_ptr<SimpleBuffer> result = solver.result();
    printf("Solved\n");
    printf("About to save\n");
    result->save("sfsOutput.imagedump");
    result->savePNG("sfsOutput", 150.0f);
    result->savePLYMesh("sfsOutput.ply");
    printf("Save\n");

	return 0;
}
