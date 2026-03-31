#!/bin/bash
# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

# Setup Loopback ALSA devices on i.MX board to allow Wav injections to be captured by eIQ GenAI Flow

# Determine configuration file location
if [ "$HOME" = "/root" ]; then
    CONFIG_FILE="/root/.asoundrc"
else
    CONFIG_FILE="$HOME/.asoundrc"
fi

echo "Creating ALSA configuration at: $CONFIG_FILE"

# Load loopback module
echo "Loading snd-aloop kernel module..."
modprobe snd-aloop

# Make it load on boot
if ! grep -q "snd-aloop" /etc/modules 2>/dev/null; then
    echo "snd-aloop" >> /etc/modules
    echo "Added snd-aloop to /etc/modules for auto-load on boot"
fi

# Create .asoundrc
cat > "$CONFIG_FILE" << 'EOF'
# ============================================
# Virtual Audio Devices for eIQ GenAI Flow Testing
# ============================================
# Loopback pairs:
#   Play to device 1 → Capture from device 0
# ============================================

# Capture endpoint (device 0) - for eiq_genai_flow
pcm.fake_capture {
    type plug
    slave {
        pcm "hw:Loopback,0,0"
        format S32_LE
        rate 16000
        channels 2
    }
    hint {
        show on
        description "Virtual Loopback Capture (Device 0)"
    }
}

# Injection endpoint (device 1) - for aplay
pcm.fake_input {
    type plug
    slave {
        pcm "hw:Loopback,1,0"
        format S32_LE
        rate 16000
        channels 2
    }
    hint {
        show on
        description "Virtual Loopback Injection (Device 1)"
    }
}

EOF

echo "Configuration created successfully!"
echo ""
echo "Verifying setup:"
echo "----------------"

# Verify loopback is loaded
if lsmod | grep -q snd_aloop; then
    echo "✓ snd-aloop module loaded"
else
    echo "✗ snd-aloop module NOT loaded"
fi

# Verify configuration file
if [ -f "$CONFIG_FILE" ]; then
    echo "✓ Configuration file exists: $CONFIG_FILE"
else
    echo "✗ Configuration file NOT found"
fi

# Verify ALSA recognizes the devices
echo ""
echo "Available virtual devices:"
aplay -L | grep -A1 "fake_"

echo ""
echo "Setup complete! You can now use:"
echo "  python eiq_genai_flow.py --capture-device plughw:CARD=Loopback --input-mode kasr"
echo ""
echo "To inject audio:"
echo "  aplay -D fake_input tests/data/question_001.wav"
