# HSV Calibration

Wth `HSV Calibration`, get the accurate `HSV Upper and Lower Range` from the `image`,`webcam`, and ,`mobile camera` (using IPCam) which can be used for `color detection`, `object tracking` and much more.

# Detect Colors Using:
- `Local Images` (Provide *url*)
- `Webcam` 
- `Mobile Camera` (Provide *IP address* using IPCam)

# Examples

1). `Images`
```sh
from hsv_calibration import ProcessImage
ProcessImage.process_image("C:\Downloads\image.jpg")   // (r"C:\Downloads\image.jpg")
```
2). `Webcam`
```sh
NOTE: Please use webcam in your project with (cv2.CAP_DSHOW): cv2.VideoCapture(0,cv2.CAP_DSHOW)

from hsv_calibration import ProcessImage
ProcessImage.process_webcam()
```
3). `Mobile Camera (Use IPCam to get the IP Address)`
```sh
from hsv_calibration import ProcessImage
ProcessImage.process_mobile_camera("192.168.2.15:8080")
```

# New Features!
- `Brightness`
- `Darkness`
- `Blur`
    - `Box Blur`
    - `Gaussian Blur`
    - `Median Blur`
    - `Bilateral Blur`
- `Hue (Low and High)`
- `Saturation (Low and High)`
- `Value (Low and High)`
- `Erosion`
- `Dilation`
- `Edge Detection using Canny`

# Why did I build this?

> To recognize or detect objects or colors in the image, OpenCV requires `HSV lower and upper range` to detect the color.
> This is time consuming as to get the accurate range, we go with hit nd trial method.
> With this library, you can easily get the accurate range and best results using other features as well.


# Requirements/Dependencies
| Requirements | Install |
| ------ | ------ |
| numpy | ` pip install numpy ` |
| opencv | ` pip install opencv-python ` |
| requests | ` pip install requests` |
| hsv-calibration | ` pip install hsv-calibration` |
        
# Installation

```sh 
pip install hsv-calibration 
```

[PyPi](https://pypi.org/project/hsv-calibration/)

# Feedback and Suggestions
For feedback and suggestions, please email me on `abhisavaliya01@gmail.com`

# About Author

`Abhi Savaliya` | `abhisavaliya01@gmail.com` | [GitHub](https://github.com/abhisavaliya) | [LinkedIn](https://www.linkedin.com/in/abhisavaliya/)
