FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI += " file://0001-AIR-12578-1-arm64-dts-Add-specific-neutron-dts-for-i.patch \
             file://0002-AIR-12578-2-neutron-Supports-allocating-memory-from-.patch \
             file://0003-neutron-move-neutron-cma-support-to-imx95-19x19-evk-.patch \
             file://0004-neutron-adjust-CMA-for-i.MX95-15x15-evk.patch \
            "

KERNEL_DEVICETREE:append:use-nxp-bsp = " \
    freescale/${KERNEL_DEVICETREE_BASENAME}-neutron.dtb \
    freescale/${KERNEL_DEVICETREE_BASENAME}-adv7535-ap1302-neutron.dtb \
"

KERNEL_DEVICETREE_INSTALL:append:use-nxp-bsp = " \
    freescale/${KERNEL_DEVICETREE_BASENAME}-neutron.dtb \
    freescale/${KERNEL_DEVICETREE_BASENAME}-adv7535-ap1302-neutron.dtb \
"
