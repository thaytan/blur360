#include "config.h"
#include "gst-equirect-blur.h"

#ifdef G_OS_UNIX
#include <glib-unix.h>
#elif defined (G_OS_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#endif

using namespace cv;
using namespace std;

struct BlurData {
  GstBin *pipeline;
  GstElement *src;
  GstElement *mux;
};

static bool draw_over_faces;
static String models_dir;
struct BlurData blur_data;

GMainLoop *loop = NULL;

#if defined(G_OS_UNIX) || defined(G_OS_WIN32)
static guint signal_watch_intr_id;

static gboolean
intr_handler (gpointer user_data)
{
  if (blur_data.src) {
    g_print ("Stopping. Sending EOS (This can take a long time while it drains queued frames)\n");
    gst_element_send_event ((GstElement *)blur_data.src, gst_event_new_eos());
  } else {
    g_print ("Stopping\n");
    g_main_loop_quit (loop);
  }
  return G_SOURCE_REMOVE;
}

#if defined(G_OS_WIN32)
static BOOL WINAPI
w32_intr_handler (DWORD dwCtrlType)
{
  intr_handler (NULL);
  SetConsoleCtrlHandler (w32_intr_handler, FALSE);
  return TRUE;
}
#endif
#endif


static gboolean
msg_handler (GstBus * bus, GstMessage * message, gpointer data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      g_print ("End Of Stream. Finished\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      GError *err = NULL;
      gchar *debug;
      gchar *name = gst_object_get_path_string (GST_MESSAGE_SRC (message));

      gst_message_parse_error (message, &err, &debug);
      g_print ("ERROR: from element %s: %s\n", name, err->message);
      if (debug) {
        g_print ("Additional debug info:\n%s\n", debug);
      }

      g_clear_error (&err);
      g_free (debug);
      g_free (name);

      g_main_loop_quit (loop);
      break;
    }

    case GST_MESSAGE_WARNING:{
      GError *err = NULL;
      gchar *debug;
      gchar *name = gst_object_get_path_string (GST_MESSAGE_SRC (message));

      gst_message_parse_warning (message, &err, &debug);
      g_print ("WARNING: from element %s: %s\n", name, err->message);
      if (debug) {
        g_print ("Additional debug info:\n%s\n", debug);
      }

      g_clear_error (&err);
      g_free (debug);
      g_free (name);
      break;
    }

    case GST_MESSAGE_PROGRESS:
    {
      GstProgressType type;
      gchar *code, *text;

      gst_message_parse_progress (message, &type, &code, &text);
      g_print ("Progress: (%s) %s\n", code, text);

      g_free (code);
      g_free (text);
    }
    default:
      break;
  }

  return TRUE;
}

static void
new_stream (GstElement *parse, GstPad * pad, struct BlurData *bd)
{
  GstCaps *caps = gst_pad_get_current_caps (pad);
  if (caps) {
    GstStructure *s = gst_caps_get_structure (caps, 0);
    const gchar *stream_type = gst_structure_get_name (s);
    GError* error = NULL;

    g_print ("New stream with type %s\n", stream_type);
    /* Link all new pads through multiqueue */
    GstElement* mq = gst_bin_get_by_name(bd->pipeline, "mq");
    assert(mq != NULL);

    GstPad* mq_sink = gst_element_get_request_pad(mq, "sink_%u");
    gchar* sink_name = gst_pad_get_name(mq_sink);
    gchar* src_name = g_strdup_printf("src_%s", sink_name + 5);
    GstPad* mq_src = gst_element_get_static_pad(mq, src_name);
    g_print("Linking stream through mq pads %s / %s\n", sink_name, src_name);
    g_free(sink_name);
    g_free(src_name);

    if (gst_pad_link(pad, mq_sink) != GST_PAD_LINK_OK) {
        cerr << "Error linking new stream to mq" << endl;
        GST_ELEMENT_ERROR(bd->pipeline, LIBRARY, INIT, ("Failed to construct pipeline fragment"), (NULL));
        goto done;
    }

    if (g_str_equal (stream_type, "video/x-h264")) {
      GstElement *blur_bin = gst_parse_bin_from_description ("avdec_h264 ! progressreport ! videoconvert name=video-in ! queue max-size-buffers=1 ! equirect_blur ! videoconvert ! x264enc tune=zerolatency name=enc ! h264parse", TRUE, &error);

      if (error != NULL) {
        cerr << "Error creating GStreamer pipeline: " << error->message << endl;
        GST_ELEMENT_ERROR (bd->pipeline, LIBRARY, INIT, ("Failed to create blurring filter"), (NULL));
        g_error_free (error);
        goto done;
      }

      gst_element_set_state (blur_bin, GST_STATE_PLAYING);
      gst_bin_add (bd->pipeline, blur_bin);

      GstPad *sinkpad = gst_element_get_static_pad (blur_bin, "sink");
      if (gst_pad_link (mq_src, sinkpad) != GST_PAD_LINK_OK) {
        cerr << "Error linking blur filter to source" << endl;
        GST_ELEMENT_ERROR (bd->pipeline, LIBRARY, INIT, ("Failed to create blurring filter"), (NULL));
        goto done;
      }

      GstPad *srcpad = gst_element_get_static_pad (blur_bin, "src");
      GstPad *mux_pad = gst_element_get_request_pad (bd->mux, "video_%u");
      if (gst_pad_link (srcpad, mux_pad) != GST_PAD_LINK_OK) {
        cerr << "Error linking blur filter to encoder" << endl;
        GST_ELEMENT_ERROR (bd->pipeline, LIBRARY, INIT, ("Failed to create blurring filter"), (NULL));
        goto done;
      }
    }
    else if (g_str_has_prefix(stream_type, "audio/")) {
      g_print ("  Trying to pass through audio stream\n");
      GstElement* pass_bin = gst_parse_bin_from_description("queue", TRUE, &error);
      if (error != NULL) {
          cerr << "Error creating GStreamer pipeline: " << error->message << endl;
          GST_ELEMENT_ERROR(bd->pipeline, LIBRARY, INIT, ("Failed to create audio filter"), (NULL));
          g_error_free(error);
          goto done;
      }

      gst_element_set_state(pass_bin, GST_STATE_PLAYING);
      gst_bin_add(bd->pipeline, pass_bin);

      GstPad* sinkpad = gst_element_get_static_pad(pass_bin, "sink");
      if (gst_pad_link(mq_src, sinkpad) != GST_PAD_LINK_OK) {
          cerr << "Error linking audio filter to source" << endl;
          GST_ELEMENT_ERROR(bd->pipeline, LIBRARY, INIT, ("Failed to create audio filter"), (NULL));
          goto done;
      }

      GstPad* srcpad = gst_element_get_static_pad(pass_bin, "src");
      GstPad *mux_pad = gst_element_get_request_pad (bd->mux, "audio_%u");
      if (gst_pad_link (srcpad, mux_pad) != GST_PAD_LINK_OK) {
        cerr << "Error linking audio stream to muxer. Ignoring." << endl;
      }
    }
    else {
      g_print ("  Unknown stream. Ignoring.\n");
    }

done:
    gst_caps_unref (caps);
  }
}

int main( int argc, char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{blur b||If supplied, faces are blurred rather than hidden with rectangles}"
                             "{models-dir m|" MODELS_DATADIR "|Path to PCN models}"
                             "{output-file o|output.mp4|Output file}"
                             "{@input-file|test.mp4|Input file}"
                             );
    parser.about( "\nA utility that extracts strips of images from an equirectangular source\n"
           "image into the relatively undistorted equatorial band, and then uses the OpenCV\n"
           "cv::CascadeClassifier class to detect faces and apply blur to them\n");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    gst_init(&argc, &argv);

    if (!gst_equirect_blur_register()) {
        std::cerr << "Failed to register GStreamer plugin" << endl;
        return 1;
    }

    String input_file = parser.get<String>("@input-file");
    String output_file = parser.get<String>("output-file");

    models_dir = parser.get<String>("models-dir");
    draw_over_faces = !parser.has("blur");

    loop = g_main_loop_new (NULL, FALSE);

    GError *error = NULL;
    GstElement *pipeline = gst_parse_launch ("filesrc name=src ! parsebin name=parse multiqueue use-interleave=true name=mq mp4mux name=mux ! filesink name=sink", &error);

    if (error != NULL) {
        cerr << "Error creating GStreamer pipeline: " << error->message << endl;
        g_error_free (error);
        return 1;
    }

    GstElement *src = gst_bin_get_by_name (GST_BIN (pipeline), "src");
    GstElement *parse = gst_bin_get_by_name (GST_BIN (pipeline), "parse");
    GstElement *mux = gst_bin_get_by_name (GST_BIN (pipeline), "mux");
    GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");

    if (src == NULL || parse == NULL || mux == NULL || sink == NULL) {
        cerr << "Error creating GStreamer pipeline: Failed to create elements" << endl;
        return 1;
    }

    g_object_set (src, "location", input_file.c_str(), NULL);
    g_object_set (sink, "location", output_file.c_str(), NULL);
    gst_object_unref (sink);

    blur_data.pipeline = GST_BIN (pipeline);
    blur_data.src= src;
    blur_data.mux = mux;

    g_signal_connect (parse, "pad-added", G_CALLBACK (new_stream), (gpointer) &blur_data);
    gst_object_unref (parse);

    GstBus *bus = gst_element_get_bus (pipeline);
    guint bus_watch_id = gst_bus_add_watch (bus, msg_handler, NULL);

    gst_object_unref (bus);

#ifdef G_OS_UNIX
    signal_watch_intr_id =
        g_unix_signal_add (SIGINT, (GSourceFunc) intr_handler, NULL);
#elif defined(G_OS_WIN32)
    SetConsoleCtrlHandler (w32_intr_handler, TRUE);
#endif

    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    g_main_loop_run (loop);

    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (blur_data.pipeline);
    gst_object_unref (blur_data.src);
    gst_object_unref (blur_data.mux);
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);

    return 0;
}
