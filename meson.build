project(
        'blur360', 'cpp',
        default_options: [ 'c_std=c99', 'cpp_std=c++11'],
        version: '0.1.0',
        meson_version: '>= 0.49',
)

dep_libm = meson.get_compiler('cpp').find_library('m', required: false)

dep_opencv = dependency('opencv4', required: false)
if not dep_opencv.found()
        dep_opencv = dependency('opencv')
endif

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

