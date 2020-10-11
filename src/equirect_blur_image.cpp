#include "config.h"
#include "equirect-blur-common.h"

using namespace cv;
using namespace std;

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{blur b||If supplied, faces are blurred rather than hidden with rectangles}"
                             "{models-dir m|" MODELS_DATADIR "|Path to PCN models}"
                             "{output-file o|output.jpg|Output file}"
                             "{@input-file|test.jpg|Input file}"
                             );
    parser.about( "\nA utility that extracts strips of images from an equirectangular source\n"
           "image into the relatively undistorted equatorial band, and then uses the OpenCV\n"
           "cv::CascadeClassifier class to detect faces and apply blur to them\n");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    String input_file = parser.get<String>("@input-file");
    Mat im = imread(input_file);
    if(im.data == NULL)
    {
        parser.printMessage();
        cout << "Can't open image file " << input_file << endl;
        return 1;
    }
    String output_file = parser.get<String>("output-file");

    String models_dir = parser.get<String>("models-dir");

    bool draw_over_faces = !parser.has("blur");

    /* Prepare cropped projection maps for processing */
    cv::Size image_size(im.cols, im.rows);
    float apertures[2] = { X_APERTURE, Y_APERTURE };
    std::vector<Projection> projections;
    cout << "Compiling detectors" << endl;

    #pragma omp parallel for
    for (int phi_step = 0; phi_step < (int)(M_PI/Y_STEP); phi_step++) {
      float phi_full = phi_step * Y_STEP;
      /* Calculate a phi (vertical tilt) from -M_PI/2 to M_PI/2 */
      float phi = phi_full <= M_PI/2 ? phi_full : phi_full - M_PI;
      for (float lambda = 0; lambda < 2*M_PI; lambda += X_STEP) {

          PCN *detector = new PCN(
                  false,
                 models_dir + "/PCN-1.caffemodel",
                 models_dir + "/PCN-1.prototxt",
                 models_dir + "/PCN-2.caffemodel",
                 models_dir + "/PCN-2.prototxt",
                 models_dir + "/PCN-3.caffemodel",
                 models_dir + "/PCN-3.prototxt",
                 models_dir + "/PCN-Tracking.caffemodel",
                 models_dir + "/PCN-Tracking.prototxt");

          /// detection
          detector->SetMinFaceSize(20);
          detector->SetImagePyramidScaleFactor(1.25f);
          detector->SetDetectionThresh(0.37f, 0.43f, 0.85f);
          /// tracking
          detector->SetTrackingPeriod(30);
          detector->SetTrackingThresh(0.9f);
          detector->SetVideoSmooth(true);

          Projection projection(image_size, apertures, phi, lambda, detector);

          #pragma omp critical
          projections.push_back(projection);
      }
    }

    cout << "Processing frame" << endl;

    if (!equirect_blur_process_frame(im, projections, draw_over_faces)) {
      cerr << "Processing frame failed" << endl;
      return 1;
    }

    imwrite(output_file, im);

    return 0;
}
