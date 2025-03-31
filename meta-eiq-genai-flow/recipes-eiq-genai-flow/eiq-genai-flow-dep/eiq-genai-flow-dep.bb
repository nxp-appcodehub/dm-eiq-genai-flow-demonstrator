SUMMARY = "eiq-genai-flow dependency installer"
DESCRIPTION = "A recipe that installs Python dependencies and tools for eiq-genai-flow"
LICENSE = "MIT"

inherit packagegroup

# List the runtime dependencies
RDEPENDS:${PN} = " \
    espeak-ng \
    onnxruntime (= 1.20.1) \
"

SRC_URI = ""

do_compile[noexec] = "1"
do_install[noexec] = "1"

# Create an empty package with just the dependencies
FILES:${PN} = ""
