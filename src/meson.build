equirect_blur_image_src = [
    'equirect_blur_image.cpp',
    'equirect-blur-common.cpp',
    'PCN.cpp'
]

executable('equirect-blur-image', equirect_blur_image_src,
  dependencies: [dep_libm, dep_opencv, dep_openmp],
  include_directories: configuration_inc)
