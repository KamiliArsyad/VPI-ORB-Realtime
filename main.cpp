#include <opencv2/core/version.hpp>
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

#include <iostream>
#include <sstream>

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

int main(int argc, char *argv[])
{
    return 0;
}
