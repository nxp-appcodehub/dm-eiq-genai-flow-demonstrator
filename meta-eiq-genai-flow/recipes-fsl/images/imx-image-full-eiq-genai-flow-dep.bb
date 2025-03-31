require dynamic-layers/qt6-layer/recipes-fsl/images/imx-image-full.bb

IMAGE_INSTALL:append = " alsa-lib-dev eiq-genai-flow-dep"
IMAGE_ROOTFS_EXTRA_SPACE = "10485760"

IMAGE_BOOT_FILES:append:use-nxp-bsp = " \
    ${KERNEL_DEVICETREE_BASENAME}-neutron.dtb \
    ${KERNEL_DEVICETREE_BASENAME}-adv7535-ap1302-neutron.dtb \
"
