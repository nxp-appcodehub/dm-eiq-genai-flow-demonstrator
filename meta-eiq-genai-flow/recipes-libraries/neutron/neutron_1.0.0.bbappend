FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI += " \
    file://NeutronDriver.h \
    file://NeutronFwllm.elf \
    file://libNeutronDriver.a \
"

do_install:append() {
    install -m 0644 ${UNPACKDIR}/NeutronFwllm.elf ${D}${nonarch_base_libdir}/firmware/
    cp ${UNPACKDIR}/NeutronDriver.h ${D}${includedir}/neutron/NeutronDriver.h
    cp --no-preserve=ownership -d ${UNPACKDIR}/libNeutronDriver.a ${D}${libdir}/libNeutronDriver.a

    install -m 0644 ${UNPACKDIR}/NeutronFwllm.elf ${D}${nonarch_base_libdir}/firmware/
    install -m 0644 ${UNPACKDIR}/NeutronDriver.h ${D}${includedir}/neutron/NeutronDriver.h
    install -m 0644 ${UNPACKDIR}/libNeutronDriver.a ${D}${libdir}/libNeutronDriver.a
}


FILES:${PN} += "${includedir}/neutron/NeutronDriver.h"
FILES:${PN}-dev += "${libdir}/libNeutronDriver.a"