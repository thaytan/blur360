project(
        'blur360', 'cpp',
        default_options: [ 'c_std=c99', 'cpp_std=c++11', 'buildtype=debugoptimized'],
        version: '0.1.0',
        meson_version: '>= 0.49',
)

dep_libm = meson.get_compiler('cpp').find_library('m', required: false)

dep_opencv = dependency('opencv4', required: false)
if not dep_opencv.found()
  dep_opencv = dependency('opencv', required: false)
endif
if not dep_opencv.found()
        dep_opencv = dependency('opencv', method: 'cmake', modules: ['opencv_core', 'opencv_imgcodecs', 'opencv_imgproc', 'opencv_dnn'])
endif

dep_openmp = dependency('openmp', required: false, language: 'cpp')

MODELS_DATADIR=join_paths(get_option('datadir'), 'blur360', 'models')

conf_data = configuration_data()
conf_data.set_quoted('MODELS_DATADIR', join_paths(get_option('prefix'), MODELS_DATADIR))

configure_file(
  output: 'config.h',
  configuration: conf_data
)

configuration_inc = include_directories('.')

subdir('src')
subdir('models')

