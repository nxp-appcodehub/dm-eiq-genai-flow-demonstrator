SUMMARY = "espeak-ng - Speech synthesis engine"
DESCRIPTION = "A multi-lingual open-source speech synthesis engine that provides text-to-speech functionality."
HOMEPAGE = "https://github.com/espeak-ng/espeak-ng"
LICENSE = "GPL-3.0-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=d32239bcb673463ab874e80d47fae504"

SRC_URI = "git://github.com/espeak-ng/espeak-ng.git;branch=master;protocol=https"
SRCREV = "2e9a5fccbb0095e87b2769e9249ea1f821918ecd"
PV = "1.51.0"
S = "${WORKDIR}/git"

# Inherit autotools for the build process
inherit autotools native

# Custom configuration options
EXTRA_OECONF = "--prefix=${prefix} --with-klatt=no --with-speechplayer=no --with-mbrola=no --with-extdict-ru=no --with-extdict-cmn=no --with-extdict-yue=no"

# Run autogen.sh before the configure step to generate missing files
do_configure() {
    cd ${S}
    if [ -f ./autogen.sh ]; then
        chmod +x ./autogen.sh
        ./autogen.sh
        ./configure ${EXTRA_OECONF}
    fi
}

# Build espeak-ng to generate dictionnaries
do_compile() {
    cd ${S}
    oe_runmake en
}

# Skip the installation step, the dictionnaries copy must be done in the non-native recipe
do_install[noexec] = "1"
