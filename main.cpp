#include <opencv2/core/version.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/ImageFlip.h>
#include <vpi/algo/ORB.h>

#include <bitset>
#include <cstdio>
#include <cstring> // for memset
#include <iostream>
#include <sstream>
#include <exception>
#define DEBUG 1

/**
 * This macro wraps a vpi-related statement to make sure that
 * it works or throws an error appropriately otherwise.
 * It is advisable to always use this on any small-scale
 * projects such as this one.
 */
#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

// Custom exception class for VPI errors
class VPIException : public std::exception
{
public:
    VPIException(VPIStatus status)
        : msg("VPI error: " + std::string(vpiStatusGetName(status))) {}

    const char *what() const noexcept override
    {
        return msg.c_str();
    }

private:
    std::string msg;
};

// Wrapper function for VPI calls
template <typename Func, typename... Args>
void vpiCall(Func func, Args... args)
{
    VPIStatus status = func(args...);
    if (status != VPI_SUCCESS)
    {
        throw VPIException(status);
    }
}

/**
 * Function to draw the keypoints later after we've identified the features (FAST)
 */
static cv::Mat DrawKeypoints(cv::Mat img, VPIKeypointF32 *kpts, int numKeypoints)
{
    cv::Mat output;
    img.convertTo(output, CV_8UC1);
    cvtColor(output, output, cv::COLOR_GRAY2BGR);

    if (!numKeypoints)
    {
        return output;
    }

    cv::Mat colorMap(1, 256, CV_8UC3);
    // Might seem a little unusual but this is a simple way to
    // temporarily create and destroy (free) our memspace.
    {
        cv::Mat gray(1, 256, CV_8UC1);
        for (int i = 0; i < 256; ++i)
        {
            gray.at<unsigned char>(0, i) = i;
        }
        applyColorMap(gray, colorMap, cv::COLORMAP_HOT);
    }

    for (int i = 0; i < numKeypoints; i++)
    {
        cv::Vec3b color = colorMap.at<cv::Vec3b>(rand() % 255);
        circle(output, cv::Point(kpts[i].x, kpts[i].y), 3, cv::Scalar(color[0], color[1], color[2]), -1);
    }

    return output;
}

/**
 * First argument: backend (<cpu|cuda>)
 * Second argument: how many frames are going to be recorded
 */
int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    VPIPayload orbPayload = NULL;
    VPIStream stream = NULL;

    int returnValue = 0;

    // Parse parameters
    if (argc != 3)
    {
        throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|cuda> <number of frames>");
    }

    int numOfFrames = std::stoi(argv[2]);
    VPIBackend backend = argv[1] == "cuda" ? VPI_BACKEND_CUDA : VPI_BACKEND_CPU;

// ========================
// Process frame by frame
#if DEBUG
    cv::VideoCapture inputCamera("../assets/input.mp4");
#else
    cv::VideoCapture inputCamera(0);
#endif

    if (!inputCamera.isOpened())
    {
        throw std::runtime_error("Can't open camera\n");
        return -1;
    }

    int32_t width = inputCamera.get(cv::CAP_PROP_FRAME_WIDTH);
    int32_t height = inputCamera.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = inputCamera.get(cv::CAP_PROP_FPS);

    // Prepare video output
    cv::VideoWriter writer;
    writer.open("out.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    if (!writer.isOpened())
    {
        throw std::runtime_error("cannot open video");
        return -1;
    }

    try
    {
        //      ---------------------
        VPIORBParams orbParams;
        // Create the stream that will be processed in the provided backend
        vpiCall(vpiStreamCreate, backend, &stream);
        vpiCall(vpiInitORBParams, &orbParams);
        orbParams.fastParams.intensityThreshold = 10;
        orbParams.maxFeatures = 500;
        //      ---------------------

        // Initialize a timer
        cv::TickMeter timer;
        timer.start();

        // Declare VPI objects
        VPIImage vpiFrame = NULL;
        VPIImage vpiFrameGrayScale = NULL;

        cv::Mat frame;
        inputCamera >> frame; // Fetch a new frame from camera.
        vpiCall(vpiImageCreate, frame.cols, frame.rows, VPI_IMAGE_FORMAT_U8, 0, &vpiFrameGrayScale);

        // Process each frame
        for (int i = 0; i < numOfFrames; ++i)
        {
            printf("processing frame %d\n", i);
            inputCamera >> frame; // Fetch a new frame from camera.

            // Declare VPI objects
            VPIPyramid pyrInput = NULL;
            VPIArray keypoints = NULL;
            VPIArray descriptors = NULL;

            // We now wrap the loaded image into a VPIImage object to be used by VPI.
            // VPI won't make a copy of it, so the original image must be in scope at all times.
            if (i == 0)
            {
                vpiImageCreateWrapperOpenCVMat(frame, 0, &vpiFrame);
            }
            else
            {
                vpiImageSetWrappedOpenCVMat(vpiFrame, frame);
            }

            // Create the output keypoint array.
            vpiCall(vpiArrayCreate,
                    orbParams.maxFeatures,
                    VPI_ARRAY_TYPE_KEYPOINT_F32,
                    backend | VPI_BACKEND_CPU,
                    &keypoints);

            // Create the output descriptors array.
            vpiCall(vpiArrayCreate,
                    orbParams.maxFeatures,
                    VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR,
                    backend | VPI_BACKEND_CPU,
                    &descriptors);

            // Create the payload for ORB Feature Detector algorithm
            vpiCall(vpiCreateORBFeatureDetector, backend, 10000, &orbPayload);

            // ---------------------
            // Process the frame
            // ---------------------

            // Convert to grayscale
            vpiCall(vpiSubmitConvertImageFormat, stream, backend, vpiFrame, vpiFrameGrayScale, nullptr);

            // Create the pyramid
            vpiCall(vpiPyramidCreate, frame.cols, frame.rows, VPI_IMAGE_FORMAT_U8, orbParams.pyramidLevels, 0.5,
                    backend, &pyrInput);
            vpiCall(vpiSubmitGaussianPyramidGenerator, stream, backend,
                    vpiFrameGrayScale, pyrInput, VPI_BORDER_CLAMP);

            // Detect ORB features
            vpiCall(vpiSubmitORBFeatureDetector, stream, backend, orbPayload,
                    pyrInput, keypoints, descriptors, &orbParams, VPI_BORDER_CLAMP);
            vpiCall(vpiStreamSync, stream);

            // ---------------------
            // Draw the keypoints and save the frame
            // ---------------------
            VPIArrayData outKeypointsData;
            VPIImageData imgData;
            vpiCall(vpiArrayLockData, keypoints, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outKeypointsData);
            vpiCall(vpiImageLockData, vpiFrameGrayScale, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData);

            VPIKeypointF32 *outKeypoints = (VPIKeypointF32 *)outKeypointsData.buffer.aos.data;

            cv::Mat tempImg;
            vpiCall(vpiImageDataExportOpenCVMat, imgData, &tempImg);
            cv::Mat output = DrawKeypoints(tempImg, outKeypoints, *outKeypointsData.buffer.aos.sizePointer);

            // Save the frame
            writer << output;

            // Cleanup
            vpiCall(vpiArrayUnlock, keypoints);
            vpiCall(vpiImageUnlock, vpiFrameGrayScale);

            vpiPyramidDestroy(pyrInput);
            vpiArrayDestroy(keypoints);
            vpiArrayDestroy(descriptors);
        }

        vpiImageDestroy(vpiFrame);
        vpiImageDestroy(vpiFrameGrayScale);

        // Stop the timer
        timer.stop();
        printf("Processing time per frame: %f ms\n", timer.getTimeMilli() / numOfFrames);
    }
    catch (const VPIException &e)
    {
        std::cerr << e.what() << '\n';
        returnValue = -1;
    }

    // Cleanup
    inputCamera.release();
    writer.release();
    vpiStreamDestroy(stream);

    return returnValue;
}
