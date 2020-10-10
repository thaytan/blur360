#include <math.h>
#include <opencv2/opencv.hpp>
#include "PCN.h"

/* Step around the sphere in overlapping ranges. Bands of 90deg vert (45deg at a time),
 * 360deg horizontal (180deg at a time) */
#define X_APERTURE ((float)(2.0f*M_PI))
#define Y_APERTURE ((float)(2.0*M_PI/3.0f))

#define X_STEP ((float)(X_APERTURE/2.0f))
#define Y_STEP ((float)(Y_APERTURE/2.0f))

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
    void create_subregion_map();
    cv::Mat eulerYZrotation(double lambda, double phi);
};

bool equirect_blur_process_frame(
    cv::Mat &image,
    std::vector<Projection> &projections,
    bool draw_over_faces);
