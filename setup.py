# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:34:03 2019

@author: abhis
"""

from distutils.core import setup
setup(
  name = 'hsv_calibration',         
  packages = ['hsv_calibration'],
  version = '0.7',
  license='MIT',  
  description = 'Accurate values for HSV lower and upper range',  
  author = 'Abhi Savaliya',                 
  author_email = 'abhisavaliya01@gmail.com',    
  url = 'https://github.com/abhisavaliya/hsv_calibration/',  
  download_url = 'https://github.com/abhisavaliya/hsv_calibration/archive/v0.1-beta7.tar.gz',
  keywords = ['OpenCV','HSV','Calibration','tool','edge','detection',
              'color','range','upper','lower','dilation','hue','saturation',
              'blur','box','gaussian','median','bilateral','bright',
              'dark','brightness','darkness','mask','erosion','opening',
              'closing','canny','threshold','easy','machine','learning',
              'image','recognition','classification','processing','computer',
              'vision'],
  install_requires=[           
          'requests',
          'numpy',
          'opencv-python'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Testing',
    'Topic :: Software Development :: Build Tools',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Scientific/Engineering :: Visualization',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.1',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)