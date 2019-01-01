#!/bin/bash

echo "[********** Start update of CentOS-7 base Image **********]"
#Update base OS
yum update -y
yum install -y wget
echo "[---********** End update of CentOS-7 base Image! **********---]\n\n"

echo "[********** Start install of CentOS-7 Development Tools **********]"
#Install Development Tools
yum install -y "Development Tools"
yum install -y make
# yum install -y centos-release-scl
# yum install -y devtoolset-7-gcc*
# scl enable devtoolset-7 #bash
echo "[---********** End install of CentOS-7 Development Tools **********---]\n\n"

echo "[********** Start install of git on CentOS-7 **********]"
#Install git
yum install -y git
echo "[---********** End install of git on CentOS-7 **********---]\n\n"

echo "[********** Start build-env configuration on CentOS-7 **********]"
#Setup build env
yum install -y autoconf automake libtool
yum install -y autoconf-archive
yum install -y pkgconfig
yum install -y libpng12-devel
yum install -y libjpeg-turbo-devel
yum install -y libtiff-devel

#*** Following needed if installing Training Tools ***#
#yum install -y zlib1g-devel => Not working???
#yum install -y libicu-devel => Not working???
#yum install -y libpango1.0-devel => Not working???
#yum install -y libcairo2-devel => Not working???
echo "[---********** End build-env configuration on CentOS-7 **********---]\n\n"

echo "[********** Start install of leptonica-1.74.4 on CentOS-7 **********]"
cd /home
mkdir tesseract
cd tesseract

#Get leptonica
wget http://www.leptonica.org/source/leptonica-1.74.4.tar.gz
gunzip leptonica-1.74.4.tar.gz
tar -xvf leptonica-1.74.4.tar
cd leptonica-1.74.4
mkdir build
cd build
../configure
make
make install
# make -j4 check
echo "[********** End install of leptonica-1.74.4 on CentOS-7 **********]\n\n"
