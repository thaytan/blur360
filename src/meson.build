equirect_blur_image_src = [
    'equirect_blur_image.cpp',
    'equirect-blur-common.cpp',
    'PCN.cpp'
]

executable('equirect-blur-image', equirect_blur_image_src,
  dependencies: [dep_libm, dep_opencv, dep_openmp],
  include_directories: configuration_inc)

if dep_gst.found() and dep_gstvideo.found()
  equirect_blur_video_src = [
      'equirect-blur-video.cpp',
      'equirect-blur-common.cpp',
      'gst-equirect-blur.cpp',
      'PCN.cpp'
  ]
  # Build the video processing
  executable('equirect-blur-video', equirect_blur_video_src,
    dependencies: [dep_libm, dep_opencv, dep_openmp, dep_gst, dep_gstvideo],
    include_directories: configuration_inc)
endif
