# Append file for onnxruntime recipe
FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI += " \
    file://0001-Add-Neutron-Execution-Provider-for-LLM-s-matmul.patch \
    "

DEPENDS += "neutron"

PACKAGECONFIG += " neutron"
PACKAGECONFIG[neutron] = "-Donnxruntime_USE_NEUTRON=ON, -Donnxruntime_USE_NEUTRON=OFF"

do_configure:prepend() {
    mkdir -p ${S}/onnxruntime/core/providers/neutron/
    cp ${STAGING_INCDIR}/neutron/NeutronDriver.h ${S}/onnxruntime/core/providers/neutron/platform/
    cp ${STAGING_LIBDIR}/libNeutronDriver.a ${S}/onnxruntime/core/providers/neutron/platform/
}

do_install:append() {
    # Identify the .whl file generated in the build directory
    WHL_FILE=$(find ${WORKDIR} -name "*.whl" | head -n 1)
    if [ -n "${WHL_FILE}" ]; then
        # Copy the .whl file to the appropriate location in the rootfs
        install -D -m 0644 ${WHL_FILE} ${D}/root/$(basename ${WHL_FILE})
    else
        bbwarn "No .whl file found in ${WORKDIR}."
    fi
}

FILES:${PN} += "/root/*.whl"
