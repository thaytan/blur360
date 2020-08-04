equirect_blur_src = [
    'equirect_blur.cpp',
    'PCN.cpp'
]

executable('equirect-blur', equirect_blur_src,
  dependencies: [dep_libm, dep_opencv, dep_openmp],
  include_directories: configuration_inc)
