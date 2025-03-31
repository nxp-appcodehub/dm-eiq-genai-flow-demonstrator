SUMMARY = "espeak-ng - Speech synthesis engine"
DESCRIPTION = "A multi-lingual open-source speech synthesis engine that provides text-to-speech functionality."
HOMEPAGE = "https://github.com/espeak-ng/espeak-ng"
LICENSE = "GPL-3.0-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=d32239bcb673463ab874e80d47fae504"

SRC_URI = "git://github.com/espeak-ng/espeak-ng.git;branch=master;protocol=https"
SRCREV = "2e9a5fccbb0095e87b2769e9249ea1f821918ecd"
PV = "1.51.0"
S = "${WORKDIR}/git"

# Build dependencies
DEPENDS = "libsndfile1 espeak-ng-native"

# Inherit autotools for the build process
inherit autotools

# Custom configuration options
EXTRA_OECONF = "--prefix=${prefix} --with-klatt=no --with-speechplayer=no --with-mbrola=no --with-extdict-ru=no --with-extdict-cmn=no --with-extdict-yue=no"

# Run autogen.sh before the configure step to generate missing files
do_configure:prepend() {
    cd ${S}
    if [ -f ./autogen.sh ]; then
        chmod +x ./autogen.sh
        ./autogen.sh --host=aarch64-poky-linux
        ./configure --host=aarch64-poky-linux ${EXTRA_OECONF}
    fi
}

# Build espeak-ng only
do_compile() {
    cd ${S}
    oe_runmake src/espeak-ng
}


NATIVE_DATA_PATH = "${WORKDIR}/../../../x86_64-linux/espeak-ng-native/${PV}/git/espeak-ng-data"

# Ensure that the required files are installed to the appropriate destination
do_install() {
    cd ${S}
    oe_runmake install-binPROGRAMS DESTDIR=${D}
    oe_runmake install-data-hook DESTDIR=${D}
    # Get data generated from native espeak-ng executable
    cp -r ${NATIVE_DATA_PATH}/* ${D}${datadir}/espeak-ng-data/

    # Ensure that all necessary files are copied correctly
    install -d ${D}${datadir}/espeak-ng-data
    chown -R root:root ${D}${datadir}/espeak-ng-data
}

# Specify the paths for binary and development files
FILES:${PN} += "${bindir}/espeak-ng \
                ${datadir}/espeak-ng-data/* \
                ${libdir}/*"

# Skip the QA check for files not shipped, specific to the vim directory
INSANE_SKIP:${PN} += "installed-vim-syntax files-not-shipped"
