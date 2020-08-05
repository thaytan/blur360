#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <opencv2/opencv.hpp>
#include "PCN.h"

using namespace std;
using namespace cv;

/* Step around the sphere in overlapping ranges. Bands of 90deg vert (45deg at a time),
 * 360deg horizontal (180deg at a time) */
#define X_APERTURE ((float)(2.0f*M_PI))
#define Y_APERTURE ((float)(M_PI/2.0f))

#define X_STEP ((float)(X_APERTURE/2.0f))
#define Y_STEP ((float)(Y_APERTURE/2.0f))

#define DEG2RAD(d) (d)*M_PI/180.0f
#define RAD2DEG(r) 180.0f*(r)/M_PI

#define ABS(x) ((x) < 0 ? -(x) : (x))

static Mat eulerYZrotation(double lambda, double phi)
{
    // Calculate rotation about Y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(lambda),  0, sin(lambda),
               0,            1, 0,
               -sin(lambda), 0, cos(lambda)
               );

    // Calculate rotation about Z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(phi), -sin(phi), 0,
               sin(phi), cos(phi),  0,
               0,        0,         1);

    return R_y * R_z;
}

static Vec2f calculate_source_uv(double u, double v, Mat& rot_mat)
{
    /* Convert to cartesian for rotation */
    Vec3d target_xyz;
    target_xyz[0] = -sin(u)*cos(v);
    target_xyz[1] = sin(u)*sin(v);
    target_xyz[2] = cos(u);

    Mat source_xyz = rot_mat * target_xyz;

    Vec2d source_uv;
    source_uv[0] = atan2(source_xyz.at<double>(1), -source_xyz.at<double>(0));
    source_uv[1] = acos(source_xyz.at<double>(2));

    if(source_uv[0] < 0)
        source_uv[0] += 2*M_PI;
    else if(source_uv[0] >= 2*M_PI)
        source_uv[0] -= 2*M_PI;

    return source_uv;
}

static Vec2f calculate_source_xy(double u, double v, Mat& rot_mat, int x_offset, int y_offset, int in_width, int in_height)
{
    Vec2f source_uv = calculate_source_uv(u, v, rot_mat);

    Vec2f src_pixel;

    src_pixel[0] = in_width*(source_uv[0])/(2*M_PI) + x_offset;
    src_pixel[1] = in_height*(source_uv[1])/M_PI + y_offset;

    return src_pixel;
}

struct Projection
{
    cv::Size equ_size;
    /* cropped X/Y aperture */
    float cropped_aperture[2];

    /* Target phi/lambda rotation for this projection */
    float phi;
    float lambda;
    cv::Mat p2eRot; /* Rotation matrix from cropped view to the source equirectangular projection */
    cv::Mat2f e2pMap; /* Mapping from equirectangular view to this cropping */

    std::vector<cv::Rect> faces; /* ROI rects in the cropped view */

    PCN *detector;

    Projection(cv::Size &im_size, float cropped_aperture[2], float phi, float lambda, PCN *detector) {
        this->equ_size = im_size;
        this->cropped_aperture[0] = cropped_aperture[0];
        this->cropped_aperture[1] = cropped_aperture[1];

        this->phi = phi;
        this->lambda = lambda;
        this->p2eRot = eulerYZrotation(phi, lambda);

        this->detector = detector;

        this->create_subregion_map();
    }
private:
    void create_subregion_map()
    {
        int x, y;
        int in_width = this->equ_size.width;
        int in_height = this->equ_size.height;
        int tmp_width = (int)round(in_width*this->cropped_aperture[0]/(2*M_PI));
        int tmp_height = (int)round(in_height*this->cropped_aperture[1]/M_PI);

        double u,v;

        this->e2pMap = Mat(tmp_height, tmp_width, CV_32FC2);

#if 0
        cout << "Creating map phi=" << projection.phi << " lambda=" << projection.lambda << endl;
        cout << "in WxH " << in_width << " x " << in_height << endl;
        cout << "target WxH " << tmp_width << " x " << tmp_height << endl;
#endif

        for (y = 0; y < tmp_height; y++) {
          for (x = 0; x < tmp_width; x++) {
              /* Calculate the U/V lat/long of the target cropped pixel */
              double x_h = (double)(x) / (tmp_width-1) - 0.5;
              double y_h = (double)(y) / (tmp_height-1) - 0.5;

              /* Scale to radians in the uncropped equirectangular frame */
              v = x_h * this->cropped_aperture[0] + M_PI;
              u = y_h * this->cropped_aperture[1] + M_PI/2;

              this->e2pMap.at<Vec2f>(y,x) = calculate_source_xy(u, v, this->p2eRot, 0, 0, in_width, in_height);
          }
        }
    }


};

/* Given an ROI on the full source image, in image coordinates,
 * calculate a map from the cropped projection into the full image */
static cv::Mat
create_roi_map_to_equ(Projection &projection, Mat &image, Mat &tmp_image, cv::Rect &roi)
{
    int x, y;
    int in_width = image.cols;
    int in_height = image.rows;
    int tmp_width = tmp_image.cols;
    int tmp_height = tmp_image.rows;

    double u,v;

    cv::Mat ret = Mat(roi.height, roi.width, CV_32FC2);
    cv::Mat e2pRot = projection.p2eRot.t();

    /* Offset for cropped image x/y */
    int x_offset = -(in_width-tmp_width)/2;
    int y_offset = -(in_height-tmp_height)/2;

    for (y = 0; y < roi.height; y++) {
      for (x = 0; x < roi.width; x++) {
          /* Calculate the U/V lat/long of the ROI pixel in the source frame */
          float x_h = (float)(x + roi.x) / (in_width-1) - 0.5;
          float y_h = (float)(y + roi.y) / (in_height-1) - 0.5;

          /* Convert to radians */
          v = x_h * 2*M_PI + M_PI;
          u = y_h * M_PI + M_PI/2;

          // Rotated UV in cropped image
          Vec2f target = calculate_source_xy(u, v, e2pRot, x_offset, y_offset, in_width, in_height);
          ret.at<Vec2f>(y,x) = target;

#if 0
          cout << "x " << x + roi.x << ", y " << y + roi.y <<
              " U = " << u << " (" << RAD2DEG(u) << ") V = " << v << " (" << RAD2DEG(v) << ") -> x " <<
              target[0] << " y " << target[1] << endl;
#endif
      }
    }

    return ret;
}
static void
extract_subregion(Projection &projection, Mat &image, Mat &tmp_image)
{
    // cout << "subregion size " << tmp_image.cols << " x " << tmp_image.rows << endl;
    cv::remap(image, tmp_image, projection.e2pMap, cv::noArray(), INTER_LINEAR, BORDER_WRAP);
}

static cv::Rect
blur_face(Projection &projection, cv::Mat img, Window face, bool draw_over_faces)
{
    /* Calculate and extract a bounding rectangle around the
     * (rotated) face and extract it as a ROI from the cropped
     * frame for blurring and later remapping into the source */
    /* Expand face rect by 25% each side */
    float face_padding = face.width/4.0;

    float x1 = face.x - face_padding;
    float y1 = face.y - face_padding;
    float x2 = face.x + face.width + face_padding - 1;
    float y2 = face.y + face.width + face_padding - 1;
    float centerX = (x1 + x2) / 2;
    float centerY = (y1 + y2) / 2;

    float angle_rad = DEG2RAD(face.angle);
    float dst_size = (face.width + 2 * face_padding);
    float rot_size = dst_size * (ABS(sin(angle_rad)) + ABS(cos(angle_rad)));
    int dst_size_pixels = (int)(ceil(dst_size));

    cv::Point2f srcTriangle[4]; /* 4th vertex is just for debug */
    cv::Point2f dstTriangle[4]; /* 4th vertex is just for debug */

    srcTriangle[0] = RotatePoint(x1, y1, centerX, centerY, (float)face.angle);
    srcTriangle[1] = RotatePoint(x1, y2, centerX, centerY, (float)face.angle);
    srcTriangle[2] = RotatePoint(x2, y2, centerX, centerY, (float)face.angle);
    srcTriangle[3] = RotatePoint(x2, y1, centerX, centerY, (float)face.angle);
    dstTriangle[0] = cv::Point(0, 0);
    dstTriangle[1] = cv::Point(0, dst_size_pixels - 1);
    dstTriangle[2] = cv::Point(dst_size_pixels - 1, dst_size_pixels - 1);
    dstTriangle[3] = cv::Point(dst_size_pixels - 1, 0);

    // cout << "Face x " << face.x << " y " << face.y << " angle " << face.angle << " (rad " << angle_rad << ") w " << face.width << " size " << rot_size << endl;

    /* Find bounding box of the source area */
    float min_x,min_y;

    if (face.angle > 0 && face.angle <= 90) {
      min_x = srcTriangle[0].x;
      min_y = srcTriangle[3].y; // srcTriangle[1].y - rot_size;
    }
    else if (face.angle > 90 && face.angle <= 270) {
      min_x = srcTriangle[3].x; // srcTriangle[1].x - rot_size;
      min_y = srcTriangle[2].y;
    }
    else if (face.angle < 0 && face.angle >= -90) {
      min_x = srcTriangle[1].x;
      min_y = srcTriangle[0].y;
    }
    else { /* -90 to -180 */
      min_x = srcTriangle[2].x;
      min_y = srcTriangle[1].y;
    }
    int max_x_pix = (int)ceil(min_x + rot_size);
    int max_y_pix= (int)ceil(min_y + rot_size);
    int min_x_pix = (int)floor(min_x);
    int min_y_pix = (int)floor(min_y);

    /* Calculate ROI to extract from the main image */
    min_x_pix = CLAMP (min_x_pix, 0, img.cols-1);
    max_x_pix = CLAMP (max_x_pix, 0, img.cols-1);
    min_y_pix = CLAMP (min_y_pix, 0, img.rows-1);
    max_y_pix = CLAMP (max_y_pix, 0, img.rows-1);

    for (int i = 0; i < 4; i++) {
        //cout << "Face quad " << i << " x " << srcTriangle[i].x << " y " << srcTriangle[i].y << endl;
        srcTriangle[i].x -= min_x;
        srcTriangle[i].y -= min_y;
    }

    cv::Rect roi = cv::Rect(min_x_pix, min_y_pix, max_x_pix-min_x_pix, max_y_pix-min_y_pix);

    //cout << "ROI x " << roi.x << " y < " << roi.y << " w " << roi.width << " h " << roi.height << endl;

    cv::Mat rotMat = cv::getAffineTransform(srcTriangle, dstTriangle);
    cv::Mat crop_roi = img(roi);
    cv::Mat face_img;

#if 0
    /* Draw quads around the whole padded area, and just around the face */
    cv::rectangle(face_img,cv::Point(face_padding,face_padding),
            cv::Point(face_padding+face.width-1,face_padding+face.width-1),cv::Scalar(255,64,64),1);

    cv::line(face_img, dstTriangle[0], dstTriangle[1], RED, 3);
    cv::line(face_img, dstTriangle[1], dstTriangle[2], BLUE, 3);
    cv::line(face_img, dstTriangle[2], dstTriangle[3], GREEN, 3);
    cv::line(face_img, dstTriangle[3], dstTriangle[0], CYAN, 3);
#elif 1
    if (draw_over_faces) {
        /* Draw grey rectangle to obscure the face */
        face_img = cv::Mat(dst_size_pixels, dst_size_pixels, img.type());
        cv::rectangle(face_img,cv::Point(0,0),cv::Point(dst_size_pixels-1, dst_size_pixels-1),cv::Scalar(64,64,64),-1);
    } else {
        /* blur the face */
        cv::warpAffine(crop_roi, face_img, rotMat, cv::Size(dst_size_pixels, dst_size_pixels));
        cv::GaussianBlur(face_img, face_img, cv::Size(31, 31), 10);
    }
#elif 0
    /* Draw GREEN rectangle to obscure the face */
    cv::rectangle(face_img,cv::Point(0,0),cv::Point(face.width-1,face.width-1),cv::Scalar(64,255,64),-1);
#endif

    /* Warp blurred/covered picture back into the source orientation */
    cv::warpAffine(face_img, crop_roi, rotMat, cv::Size(crop_roi.rows, crop_roi.cols),
            cv::WARP_INVERSE_MAP|cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    //imshow("Face", crop_roi);
    //imshow("Face", face_img);
    //waitKey(0);

    /* Return the rect that needs copying back into the original equirect projection */
    return roi;
}


// We have faces to project back to the full frame
// For each face, calculate bounding rectangles in the
// equirect frame and generate a map back from the face ROI
// to it. The reprojection may cross the edges of the image
// and gets complicated
static void
project_faces_to_full_frame(Projection &projection, cv::Mat &equ_image,
    cv::Mat &cropped_image)
{
    std::vector<cv::Rect> rects; /* ROI rects in the source frame */

    for (size_t f = 0; f < projection.faces.size(); f++)
    {
        cv::Rect roi = projection.faces[f];
        cv::Point2f srcQuad[4];
        cv::Point2f dstQuad[4];
        srcQuad[0] = cv::Point(roi.x, roi.y);
        srcQuad[1] = cv::Point(roi.x, roi.y + roi.height);
        srcQuad[2] = cv::Point(roi.x + roi.width, roi.y + roi.height);
        srcQuad[3] = cv::Point(roi.x + roi.width, roi.y);
        int min_x, min_y, max_x, max_y;

        // cout << "Face quad: " << endl;
        for (int i = 0; i < 4; i++) {
            int y = (int)round(srcQuad[i].y);
            int x = (int)round(srcQuad[i].x);

            Vec2f p = projection.e2pMap.at<Vec2f>(y,x);
            dstQuad[i] = Point2f(p[0], p[1]);
            //cout << "  vertex " << i << " from " << x << ", " << y << " src image " << p[0] << ", " << p[1] << endl;
        }

        min_x = (int)floor(MIN(dstQuad[0].x, dstQuad[1].x));
        max_x = (int)ceil(MAX(dstQuad[2].x, dstQuad[3].x));

        min_y = (int)floor(MIN(dstQuad[0].y, dstQuad[3].y));
        max_y = (int)ceil(MAX(dstQuad[1].y, dstQuad[2].y));

#if 0
        cv::line(equ_image, dstQuad[0], dstQuad[1], RED, 3);
        cv::line(equ_image, dstQuad[1], dstQuad[2], BLUE, 3);
        cv::line(equ_image, dstQuad[2], dstQuad[3], GREEN, 3);
        cv::line(equ_image, dstQuad[3], dstQuad[0], CYAN, 3);
#endif

        /* The destination quad may cross edges and need to be split into up to 4 sub-quads and
         * remapped */
        if (min_x > max_x) {
            if (min_y > max_y) {
                /* Crossed both right and bottom edges - 4 quads */
                rects.push_back(cv::Rect(min_x, min_y, equ_image.cols - 1 - min_x, equ_image.rows - 1 - min_y));
                rects.push_back(cv::Rect(0, min_y, max_x, equ_image.rows - 1 - min_y));
                rects.push_back(cv::Rect(min_x, 0, equ_image.cols - 1 - min_x, max_y));
                rects.push_back(cv::Rect(0, 0, max_x, max_y));
            }
            else {
                /* Crossed right edge */
                rects.push_back(cv::Rect(min_x, min_y, equ_image.cols - 1 - min_x, max_y-min_y));
                rects.push_back(cv::Rect(0, min_y, max_x, max_y-min_y));
            }
        }
        else if (min_y > max_y) {
            /* Crossed bottom edge */
            rects.push_back(cv::Rect(min_x, min_y, max_x-min_x, equ_image.rows - 1 - min_y));
            rects.push_back(cv::Rect(min_x, 0, max_x-min_x, max_y));
        } else {
            /* Just one quad */
            rects.push_back(cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y));
        }
    }

#if 0
    for (int i = 0; i < rects.size(); i++) {
        cv::Rect roi = rects[i];
        cout << "ROI x " << roi.x << " y " << roi.y << " w " << roi.width << " h " << roi.height << endl;
    }
#endif

    for (size_t i = 0; i < rects.size(); i++) {
      cv::Rect &rect = rects[i];

      if (rect.width == 0 || rect.height == 0)
          continue;

      cv::Mat map = create_roi_map_to_equ(projection, equ_image, cropped_image, rect);
      cv::Mat image_roi = equ_image(rect);
      cv::remap(cropped_image, image_roi, map, cv::noArray(), INTER_LINEAR, BORDER_TRANSPARENT);
    }
}

static void
process_frame(String output_file,
    Mat &image,
    std::vector<Projection> &projections,
    bool draw_over_faces)
{
    /*
     * Sweep the sphere in steps, calculating a centre
     * λ and φ and extract sub-images that should allow face
     * recognition to work at latitudes away from the equator
     */
    int in_width = image.cols;
    int in_height = image.rows;

    /* Calculate a temporary image size that matches the target aperture */
    int tmp_width = (int)round(in_width * X_APERTURE/(2*M_PI));
    int tmp_height = (int)round(in_height * Y_APERTURE/(M_PI));

    Mat tmp_image(tmp_height, tmp_width, image.type());

    for (size_t n = 0; n < projections.size(); n++) {
      Projection &p = projections[n];

      //cout << "Region phi=" << p.phi << " lambda=" << p.lambda << endl;

      extract_subregion(p, image, tmp_image);

#if 0
      imshow("Cropped frame", tmp_image);
      waitKey(0);
#endif

      // Detect faces in this sub-image
      std::vector<Window> faces = p.detector->Detect(tmp_image);

      // Extract faces and blur into the cropped image
      if (faces.size()) {
          //cout << "Detected " << faces.size() << " faces" << endl;
          for (size_t j = 0; j < faces.size(); j++)
          {
              p.faces.push_back(blur_face(p, tmp_image, faces[j], draw_over_faces));
              //DrawFace(tmp_image, faces[j]);
              //drawpoints(tmp_image, faces[j]);
          }

#if 0
          imshow("Region", tmp_image);
          std::stringstream fname;
          fname << "img" << n << ".jpg";
          imwrite(fname.str().c_str(), tmp_image);
          waitKey(0);
#endif
          // Project blurred areas back to the full frame
          project_faces_to_full_frame(p, image, tmp_image);
      }
    }

    imwrite(output_file, image);
#if 0
    imshow("Source", image);
    waitKey(0);
#endif
}

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

          PCN *detector = new PCN(models_dir + "/PCN.caffemodel",
                 models_dir + "/PCN-1.prototxt",
                 models_dir + "/PCN-2.prototxt",
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

          projections.push_back(projection);
      }
    }

    cout << "Processing frame" << endl;
    process_frame(output_file, im, projections, draw_over_faces);

    return 0;
}
