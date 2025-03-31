FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

# Ensure development headers are installed
do_install:append() {
    install -d ${D}${includedir}/alsa
    install -m 0644 ${S}/include/*.h ${D}${includedir}/alsa
}

# Make sure the headers are included in the dev package
FILES:${PN}-dev += "${includedir}/alsa/*.h"