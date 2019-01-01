echo "[********** Start update of CentOS-7 base Image **********]"
#Update base OS
yum update -y
yum install -y wget
echo "[---********** End update of CentOS-7 base Image! **********---]\n\n"

echo "[********** Start install of CentOS-7 Development Tools **********]"
#Install Development Tools
yum install -y "Development Tools"
yum install -y make
yum install -y centos-release-scl
yum install -y devtoolset-7-gcc*
scl enable devtoolset-7 #bash
echo "[---********** End install of CentOS-7 Development Tools **********---]\n\n"

echo "[********** Start install of git on CentOS-7 **********]"
#Install git
yum install -y git
echo "[---********** End install of git on CentOS-7 **********---\n\n]"

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

echo "[********** Start install of tesseract-4.0.0-beta.1 on CentOS-7 **********]"
#Let pkg-config know where Leptonica is installed
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/
cd /home/tesseract
wget https://github.com/tesseract-ocr/tesseract/archive/4.0.0-beta.1.tar.gz
gunzip 4.0.0-beta.1.tar.gz
tar -xvf 4.0.0-beta.1.tar
rm 4.0.0-beta.1.tar
cd tesseract-4.0.0-beta.1/

./autogen.sh
#./configure
LIBLEPT_HEADERSDIR=/usr/local/include/leptonica ./configure --prefix=/usr/local/ --with-extra-libraries=/usr/local/lib
make
make install
ldconfig
echo "[********** End install of tesseract-4.0.0-beta.1 on CentOS-7 **********]\n\n"
