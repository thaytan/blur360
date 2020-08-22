
#ifndef __GST_EQUIRECT_BLUR_H__
#define __GST_EQUIRECT_BLUR_H__

#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>

#include "equirect-blur-common.h"

G_BEGIN_DECLS

#define GST_TYPE_EQUIRECT_BLUR (gst_equirect_blur_get_type())
G_DECLARE_FINAL_TYPE (GstEquirectBlur, gst_equirect_blur,
    GST, EQUIRECT_BLUR, GstVideoFilter);

struct _GstEquirectBlur {
  GstVideoFilter parent;

  gint width, height;

  gboolean update_projections;
  std::vector<Projection> projections;
  cv::Mat cvMat;

  gboolean draw_over_faces;
  gchar *models_dir;
};

struct _GstEquirectBlurClass {
    GstVideoFilterClass parent;
};

gboolean gst_equirect_blur_register(void);

G_END_DECLS

#endif
