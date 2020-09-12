# Blur360

This project aims to provide face blurring and obscuring for 360 images and videos in equirectangular projection. Equirectangular projection presents some special challenges
for face detection and blurring due to the strong distortion away from the equator.

This project detects faces in several re-projections of the input frame, moving an
area of interest into the equatorial zone on each pass, finding faces and then re-projecting the obscured version back to the original frame for output.

## Face detection

Face detection is done using the PCN model from https://github.com/MagicCharles/FaceKit ported to use OpenCV's DNN
module instead of Caffe.

## Current status

At the moment, blurring and face overdrawing is supported for single images:

```
./bin/equirect-blur-image -m=models -o=output.jpg Input.jpg
```

and for some video files:
```
./bin/equirect-blur-video -m=models -o=output.mp4 Input.mp4
```

By default, this will draw grey rectangles to completely obscure detected faces. To blur faces instead, use the `-b` command line option.

Output encoding is fixed in the source code - JPEG for images, H.264 + mp4 for video.

## Building

Compilation requires `meson` and `OpenCV`. Depending on platform you might also need `ninja`

meson: https://mesonbuild.com/Getting-meson.html
OpenCV: On Linux, install via the package manager. For other platforms try https://opencv.org/releases/

With those installed (for Linux):
```
meson build
ninja -C build
```

For Windows, you probably want to use the MSVC backend for meson.
