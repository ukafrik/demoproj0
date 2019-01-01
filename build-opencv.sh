#!/bin/bash

#Update base OS
yum update -y

#Install Development Tools
yum install -y "Development Tools"

#Install git
yum install -y git

#Setup build env
yum install -y autoconf automake libtool
yum install -y autoconf-archive
yum install -y pkgconfig
yum install -y libpng12-devel
yum install -y libjpeg-turbo-devel
yum install -y libtiff-devel
#yum install -y zlib1g-devel => Not working???
#yum install -y libicu-devel => Not working???
#yum install -y libpango1.0-devel => Not working???
#yum install -y libcairo2-devel => Not working???

#Install wget since it may not be installed on CentOS
yum install -y wget

####### Build OpenCV #######
cd /home
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.1.zip
unzip opencv.zip
cd opencv
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.1.zip
unzip opencv_contrib.zip
mkdir build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=
#make -j4
make install
ldconfig

