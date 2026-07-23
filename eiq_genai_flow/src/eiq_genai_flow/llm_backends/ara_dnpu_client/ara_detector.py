# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import subprocess
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class AraDetector:
    """Detect ARA2 discrete NPU availability and configuration."""

    def __init__(self, config):
        self.config = config
        self.m2_port_path = "/sys/bus/pci/devices"
        self.llm_base_path = "/usr/share/llm"
        self.connector_path = "/usr/share/eiq/aaf-connector"
        self.connector_host = "0.0.0.0"
        self.connector_port = 8000
        self.base_url = f"http://{self.connector_host}:{self.connector_port}"

        logger.debug("=" * 60)
        logger.debug("Initializing ARA DNPU Detector")
        logger.debug("=" * 60)

    def check_m2_port(self) -> bool:
        """Check if M.2 port is available."""
        logger.debug("Checking M.2 port availability...")
        try:
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            has_pci = "PCI bridge" in result.stdout or "Express" in result.stdout
            logger.debug(f"  M.2 port check: {'✓ FOUND' if has_pci else '✗ NOT FOUND'}")
            if has_pci:
                logger.debug("  PCIe devices detected")
            return has_pci
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"  M.2 port check: ✗ FAILED ({e})")
            return False

    def check_ara2_module(self) -> bool:
        """
        Check if ARA2 module is installed on M.2 port.

        The ARA2 appears in lspci as:
        - Vendor ID: 1e58
        - Device ID: 0002
        - Description: "Processing accelerators"
        """
        logger.debug("Checking ARA2 module presence...")

        try:
            # Check by vendor:device ID (most reliable)
            result = subprocess.run(["lspci", "-nn"], capture_output=True, text=True, timeout=5)

            # Look for the specific ARA2 vendor:device ID
            if "1e58:0002" in result.stdout:
                logger.debug("  ARA2 module: ✓ FOUND (PCI device 1e58:0002)")
                return True

            logger.debug("  ARA2 module: ✗ NOT FOUND")
            return False

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"  ARA2 module check: ✗ FAILED ({e})")
            return False

    def check_service_exists(self) -> bool:
        """Check if rt-sdk-ara2.service is installed (exists)."""
        logger.debug(f"Checking if {self.config.discrete_npu_service} exists...")
        try:
            # Try to get service status - will fail if service doesn't exist
            result = subprocess.run(
                ["systemctl", "status", self.config.discrete_npu_service], capture_output=True, text=True, timeout=5
            )

            # Check if service was found
            if "could not be found" in result.stderr or "not found" in result.stdout.lower():
                logger.debug("  Service: ✗ NOT INSTALLED")
                return False

            logger.debug("  Service: ✓ INSTALLED")
            return True

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"  Service check: ✗ FAILED ({e})")
            return False

    def get_service_state(self) -> dict:
        """Get detailed service state information."""
        state = {
            "exists": False,
            "loaded": False,
            "active": False,
            "enabled": False,
            "status": "unknown",
            "sub_state": "unknown",
        }

        try:
            # Check if service exists and get status
            result = subprocess.run(
                [
                    "systemctl",
                    "show",
                    self.config.discrete_npu_service,
                    "--property=LoadState,ActiveState,SubState,UnitFileState",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key == "LoadState":
                            state["loaded"] = value == "loaded"
                            state["exists"] = value != "not-found"
                        elif key == "ActiveState":
                            state["active"] = value == "active"
                            state["status"] = value
                        elif key == "SubState":
                            state["sub_state"] = value
                        elif key == "UnitFileState":
                            state["enabled"] = value == "enabled"

            return state

        except Exception as e:
            logger.debug(f"Failed to get service state: {e}")
            return state

    def check_rt_sdk_service(self) -> bool:
        """Check if rt-sdk-ara2.service is running (active)."""
        logger.debug(f"Checking {self.config.discrete_npu_service} service...")

        state = self.get_service_state()

        if not state["exists"]:
            logger.debug("  Service: ✗ NOT INSTALLED")
            logger.warning(f"  {self.config.discrete_npu_service} is not installed on this system")
            return False

        if not state["loaded"]:
            logger.debug("  Service: ✗ NOT LOADED")
            return False

        is_active = state["active"]
        status = state["status"]
        sub_state = state["sub_state"]

        logger.debug(f"  Service status: {status} ({sub_state}) {'✓' if is_active else '✗'}")

        if state["enabled"]:
            logger.debug("  Service enabled: ✓ (starts on boot)")
        else:
            logger.debug("  Service enabled: ✗ (manual start only)")

        return is_active

    def start_rt_sdk_service(self) -> bool:
        """
        Start the rt-sdk-ara2.service if it's not running.

        Returns:
            True if service started successfully or was already running
        """
        logger.info(f"Starting {self.config.discrete_npu_service}...")

        # First check if it exists
        state = self.get_service_state()

        if not state["exists"]:
            logger.error(f"  ✗ Service not installed: {self.config.discrete_npu_service}")
            logger.error("  Install the rt-sdk package first")
            return False

        # Check if already running
        if state["active"]:
            logger.info("  Service is already running")
            return True

        # Try to start it
        try:
            logger.info("  Attempting to start service...")
            result = subprocess.run(
                ["sudo", "systemctl", "start", self.config.discrete_npu_service],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Wait a bit and verify it started
                import time

                time.sleep(2)

                # Check if now active
                new_state = self.get_service_state()
                if new_state["active"]:
                    logger.info("  ✓ Service started successfully")
                    return True
                else:
                    logger.error("  ✗ Service failed to start")
                    logger.error(f"    Status: {new_state['status']} ({new_state['sub_state']})")
                    return False
            else:
                logger.error("  ✗ Failed to start service")
                if result.stderr:
                    logger.error(f"    Error: {result.stderr.strip()}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("  ✗ Service start timed out")
            return False
        except FileNotFoundError:
            logger.error("  ✗ systemctl command not found")
            return False
        except Exception as e:
            logger.error(f"  ✗ Failed to start service: {e}")
            return False

    def check_connector_package(self) -> bool:
        """Check if eiq-aaf-connector_x.x.deb is installed."""
        logger.debug("Checking eiq-aaf-connector package...")
        try:
            result = subprocess.run(["dpkg", "-l", "eiq-aaf-connector"], capture_output=True, text=True, timeout=5)
            is_installed = "ii" in result.stdout and "eiq-aaf-connector" in result.stdout
            logger.debug(f"  Package installed: {'✓ YES' if is_installed else '✗ NO'}")
            if is_installed:
                # Extract version
                lines = result.stdout.split("\n")
                for line in lines:
                    if "eiq-aaf-connector" in line and line.startswith("ii"):
                        parts = line.split()
                        if len(parts) >= 3:
                            logger.debug(f"  Version: {parts[2]}")
            return is_installed
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"  Package check: ✗ FAILED ({e})")
            return False

    def check_connector_running(self) -> bool:
        """Check if eiq-aaf-connector process is running (not zombie)."""
        logger.debug("Checking if aaf-connector is running...")
        try:
            result = subprocess.run(["ps", "-aux"], capture_output=True, text=True, timeout=5)

            for line in result.stdout.split("\n"):
                if "connector" in line and "aaf-connector" in line:
                    # Check if process is a zombie (Z state)
                    parts = line.split()
                    if len(parts) >= 8:
                        state = parts[7]  # STAT column
                        if "Z" in state or "<defunct>" in line:
                            logger.warning(f"  Connector process is a ZOMBIE (defunct): {line.strip()}")
                            logger.warning("  The connector crashed — needs to be restarted")
                            # Try to clean up zombie
                            self._cleanup_zombie_connector()
                            return False

                        logger.debug(f"  Connector process: ✓ RUNNING (state: {state})")
                        logger.debug(f"  Process: {line.strip()}")
                        return True

            logger.debug("  Connector process: ✗ NOT RUNNING")
            return False

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"  Connector process check: ✗ FAILED ({e})")
            return False

    def _cleanup_zombie_connector(self):
        """
        Attempt to clean up zombie connector process.
        Zombies can only be removed by their parent or by reparenting to init.
        """
        logger.info("  Attempting to clean up zombie connector process...")
        try:
            # Find the zombie PID
            result = subprocess.run(["ps", "-aux"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split("\n"):
                if "connector" in line and ("<defunct>" in line or "aaf-connector" in line):
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        state = parts[7] if len(parts) > 7 else ""
                        if "Z" in state or "<defunct>" in line:
                            logger.info(f"  Found zombie PID: {pid}")
                            # Kill parent to force zombie cleanup
                            ppid_result = subprocess.run(
                                ["ps", "-o", "ppid=", "-p", pid], capture_output=True, text=True, timeout=5
                            )
                            ppid = ppid_result.stdout.strip()
                            if ppid and ppid != "1":
                                logger.info(f"  Sending SIGCHLD to parent PID: {ppid}")
                                subprocess.run(["kill", "-CHLD", ppid], timeout=5)
                            break
        except Exception as e:
            logger.debug(f"  Zombie cleanup failed: {e}")

    def check_connector_api(self) -> bool:
        """
        Test if connector API is responding using /v1/models endpoint.

        Returns:
            True if API is accessible and responding
        """
        logger.debug(f"Testing connector API at {self.base_url}...")
        try:
            url = f"{self.base_url}/v1/models"
            logger.debug(f"  Attempting GET {url}")
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                logger.debug(f"  API response: ✓ OK (status {response.status_code})")
                try:
                    data = response.json()
                    if "data" in data:
                        model_count = len(data["data"])
                        logger.debug(f"  Models available: {model_count}")
                except Exception:
                    logger.debug(f"  Response: {response.text[:100]}")
                return True
            else:
                logger.debug(f"  API response: ✗ FAILED (status {response.status_code})")
                return False
        except Exception as e:
            logger.debug(f"  API check: ✗ FAILED ({type(e).__name__}: {e})")
            return False

    def get_devices_info(self) -> dict:
        """
        Get ARA device information using /v1/devices endpoint.

        Returns:
            Dictionary with device information or None if failed
        """
        logger.debug("Fetching ARA device information...")
        try:
            url = f"{self.base_url}/v1/devices"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                logger.debug("  ✓ Retrieved device info")
                logger.debug(f"  Device count: {data.get('device_count', 0)}")

                # Log device details
                for device in data.get("devices", []):
                    device_id = device.get("device_id")
                    memory = device.get("memory")
                    models = device.get("loaded_models", [])

                    logger.debug(f"  Device {device_id}:")
                    if memory:
                        logger.debug(
                            f"    Memory: {memory.get('used_human')} / {memory.get('total_human')} "
                            f"({memory.get('utilization_percent', 0):.1f}% used)"
                        )
                    logger.debug(f"    Loaded models: {len(models)}")
                    for model in models:
                        status = "✓ READY" if model.get("ready") else "✗ NOT READY"
                        logger.debug(f"      - {model.get('name')} ({model.get('type')}): {status}")

                return data
            else:
                logger.debug(f"  ✗ Failed to get device info (status {response.status_code})")
                return None

        except Exception as e:
            logger.debug(f"  ✗ Failed to get device info: {e}")
            return None

    def check_model_loaded(self, model_name: str) -> tuple[bool, bool]:
        """
        Check if a specific model is loaded and ready using /v1/models list.
        The model appearing in /v1/models means it is loaded and ready.

        Args:
            model_name: Name of the model to check (without -ara suffix)

        Returns:
            Tuple of (is_loaded, is_ready)
        """
        logger.debug(f"Checking if model '{model_name}' is loaded via /v1/models...")
        try:
            # First try /v1/models/{model_name} endpoint
            url = f"{self.base_url}/v1/models/{model_name}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if "id" in data and data["id"] == model_name:
                    is_ready = data.get("ready", True)  # Default True if field absent
                    logger.debug(f"  Model '{model_name}': ✓ LOADED via /v1/models/{{name}}, ready={is_ready}")
                    return True, is_ready

            # Fallback: check /v1/models list
            url_list = f"{self.base_url}/v1/models"
            response_list = requests.get(url_list, timeout=5)

            if response_list.status_code == 200:
                data = response_list.json()
                models = data.get("data", [])
                for model in models:
                    model_id = model.get("id", "")
                    if model_id == model_name:
                        # Model present in list = loaded and ready
                        logger.debug(f"  Model '{model_name}': ✓ FOUND in /v1/models list → ready")
                        return True, True

                logger.debug(f"  Model '{model_name}': ✗ NOT in /v1/models list yet")
                return False, False

            logger.debug(f"  /v1/models returned status {response_list.status_code}")
            return False, False

        except Exception as e:
            logger.debug(f"  Model check failed: {e}")
            return False, False

    def wait_for_model_ready(self, model_name: str, timeout: int = 300, check_interval: int = 5) -> bool:
        """
        Wait for a specific model to be loaded and ready.
        Uses /v1/models list as primary check — model presence = ready.

        Args:
            model_name: Name of the model to wait for (without -ara suffix)
            timeout: Maximum time to wait in seconds (default: 5 minutes)
            check_interval: How often to check in seconds (default: 5 seconds)

        Returns:
            True if model is ready, False if timeout
        """
        import time

        logger.info("=" * 60)
        logger.info(f"Waiting for model '{model_name}' to be ready...")
        logger.info("This can take several minutes on first load")
        logger.info("=" * 60)

        start_time = time.time()
        elapsed = 0
        attempt = 0

        while elapsed < timeout:
            attempt += 1
            elapsed = time.time() - start_time

            is_loaded, is_ready = self.check_model_loaded(model_name)

            if is_loaded and is_ready:
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"✓ Model '{model_name}' is ready after {elapsed:.1f} seconds")
                logger.info("=" * 60)
                return True

            # Show progress every 30 seconds
            if attempt % 6 == 0:
                status = "loading" if is_loaded else "not in model list yet"
                logger.info(f"  Model status: {status} ({elapsed:.0f}s / {timeout}s)")

            time.sleep(check_interval)

        logger.error("")
        logger.error("=" * 60)
        logger.error(f"✗ Model '{model_name}' did not become ready within {timeout} seconds")
        logger.error("=" * 60)
        return False

    def start_connector(self) -> bool:
        """Attempt to start the connector if not running."""
        logger.info("Attempting to start aaf-connector...")

        # Check if already running FIRST
        if self.check_connector_running():
            logger.info("  Connector already running, skipping start")
            return True

        try:
            venv_activate = f"{self.connector_path}/venv/bin/activate"
            connector_cmd = f"{self.connector_path}/venv/bin/connector"

            if not Path(venv_activate).exists():
                logger.warning(f"  Connector venv not found at {venv_activate}")
                return False

            if not Path(connector_cmd).exists():
                logger.warning(f"  Connector binary not found at {connector_cmd}")
                return False

            logger.info(f"  Starting connector: {connector_cmd}")

            # In DEBUG mode, capture connector stdout/stderr to a log file
            connector_log_path = "/tmp/connector.log"
            if logger.isEnabledFor(logging.DEBUG):
                log_file = open(connector_log_path, "w")
                logger.debug(f"  Connector output will be captured to {connector_log_path}")
            else:
                log_file = None

            stdout_dest = log_file or subprocess.DEVNULL
            stderr_dest = log_file or subprocess.DEVNULL

            # Start connector in background
            subprocess.Popen(
                ["/bin/bash", "-c", f"source {venv_activate} && {connector_cmd}"],
                stdout=stdout_dest,
                stderr=stderr_dest,
                start_new_session=True,
            )

            # Wait and verify
            import time

            logger.info("  Waiting for connector to start...")
            for i in range(5):
                time.sleep(1)
                if self.check_connector_running():
                    logger.info("  ✓ Connector started successfully")
                    if log_file:
                        logger.info(f"  Connector logs: tail -f {connector_log_path}")
                    return True
                logger.debug(f"  Still waiting... ({i + 1}/5)")

            logger.warning("  ✗ Connector failed to start")
            return False

        except Exception as e:
            logger.error(f"  ✗ Failed to start connector: {e}")
            return False

    def get_available_llms(self) -> list[str]:
        """
        Get list of available LLMs combining filesystem scan and API query.

        Strategy:
        - Filesystem scan: works without connector running (from server_config.json)
        - API query: works when connector is running (live model list)
        - Both results are merged to get the most complete list

        Returns:
            List of model names with '-ara' suffix
        """
        llms = set()

        # --- Approach 1: Filesystem scan (works without connector) ---
        filesystem_llms = self._get_llms_from_filesystem()
        if filesystem_llms:
            logger.debug(f"  Filesystem scan found {len(filesystem_llms)} model(s): {filesystem_llms}")
            llms.update(filesystem_llms)

        # --- Approach 2: API query (works when connector is running) ---
        api_llms = self._get_llms_from_api()
        if api_llms:
            logger.debug(f"  API query found {len(api_llms)} model(s): {api_llms}")
            llms.update(api_llms)

        if llms:
            logger.debug(f"  Total unique models found: {len(llms)}")
        else:
            logger.debug("  No ARA models found via filesystem or API")

        return sorted(llms)

    def _get_llms_from_filesystem(self) -> list[str]:
        """
        Get list of LLMs available in /usr/share/llm/ that are enabled in server config.
        This works without the connector running.

        Returns:
            List of model names with '-ara' suffix
        """
        logger.debug(f"Scanning for LLMs in {self.llm_base_path}...")

        try:
            import json

            # Read server configuration
            config_path = Path(f"{self.connector_path}/server_config.json")
            if not config_path.exists():
                logger.debug(f"  Server config not found: {config_path}")
                return []

            try:
                with open(config_path, "r") as f:
                    server_config = json.load(f)
            except json.JSONDecodeError as e:
                logger.debug(f"  Failed to parse server config: {e}")
                return []

            # Get enabled models from config
            available_models = server_config.get("available_models", [])
            enabled_models = {model["name"] for model in available_models if model.get("enabled", False)}

            if not enabled_models:
                logger.debug("  No enabled models in server config")
                return []

            logger.debug(f"  Enabled models in config: {sorted(enabled_models)}")

            # Check which enabled models exist in filesystem
            llm_path = Path(self.llm_base_path)
            if not llm_path.exists():
                logger.debug(f"  LLM path does not exist: {llm_path}")
                return []

            llms = []
            for item in llm_path.iterdir():
                if item.is_dir() and item.name in enabled_models:
                    # Check for model files (.dvm is the ARA2 model format)
                    model_files = (
                        list(item.glob("*.dvm"))
                        + list(item.glob("*.safetensors"))
                        + list(item.glob("*.bin"))
                        + list(item.glob("config.json"))
                    )

                    if model_files:
                        model_name = item.name
                        llms.append(f"{model_name}-ara")
                        logger.debug(f"  Found enabled model: {model_name} ({len(model_files)} files)")
                    else:
                        logger.debug(f"  Model {item.name} is enabled but missing files")

            return sorted(llms)

        except Exception as e:
            logger.debug(f"  Filesystem LLM discovery failed: {e}")
            return []

    def _get_llms_from_api(self) -> list[str]:
        """
        Get list of LLMs from connector API.
        This works only when the connector is running.

        Returns:
            List of model names with '-ara' suffix
        """
        logger.debug("Fetching available models from connector API...")

        try:
            url = f"{self.base_url}/v1/models"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                models = []

                # Parse OpenAI-compatible response
                if "data" in data:
                    for model in data["data"]:
                        model_id = model.get("id")
                        if model_id:
                            ara_name = f"{model_id}-ara"
                            models.append(ara_name)
                            logger.debug(f"  API found model: {model_id}")

                return sorted(models)
            else:
                logger.debug(f"  API not available (status {response.status_code}), skipping API query")
                return []

        except requests.ConnectionError:
            logger.debug("  Connector API not running, skipping API query")
            return []
        except Exception as e:
            logger.debug(f"  API model query failed: {e}")
            return []

    def is_ara_available(self, auto_start_connector: bool = True) -> tuple[bool, list[str]]:
        """
        Check if ARA DNPU is available and return available LLMs.

        Args:
            auto_start_connector: Attempt to start connector if not running

        Returns:
            (is_available, list_of_llms)
        """
        logger.debug("")
        logger.debug("=" * 60)
        logger.debug("Starting ARA DNPU Detection")
        logger.debug("=" * 60)

        # Check hardware first
        has_m2_port = self.check_m2_port()
        has_ara2_module = self.check_ara2_module()

        if not has_m2_port or not has_ara2_module:
            logger.debug("")
            logger.debug("Hardware prerequisites not met:")
            if not has_m2_port:
                logger.debug("  ✗ M.2 port not available")
            if not has_ara2_module:
                logger.debug("  ✗ ARA2 module not detected")
            logger.debug("=" * 60)
            return False, []

        # Check if service exists
        if not self.check_service_exists():
            logger.debug("")
            logger.debug(f"{self.config.discrete_npu_service} is not installed")
            logger.debug("Install the rt-sdk-ara2 package to use ARA DNPU")
            logger.debug("=" * 60)
            return False, []

        # Run remaining software checks
        checks = {
            "M.2 port": has_m2_port,
            "ARA2 module": has_ara2_module,
            "connector package": self.check_connector_package(),
        }

        # Check service state
        service_state = self.get_service_state()
        service_running = service_state["active"]

        # Try to start service if not running
        if not service_running:
            logger.info(f"{self.config.discrete_npu_service} is not running, attempting to start...")
            service_running = self.start_rt_sdk_service()

        checks["rt-sdk service"] = service_running

        logger.debug("")
        logger.debug("Check Summary:")
        for check_name, result in checks.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.debug(f"  {check_name:20s}: {status}")

        # All prerequisite checks must pass
        if not all(checks.values()):
            logger.debug("")
            logger.debug("Result: ARA DNPU NOT AVAILABLE (prerequisites not met)")

            # Give helpful hints
            if not service_running:
                logger.warning("")
                logger.warning(f"The {self.config.discrete_npu_service} could not be started.")
                logger.warning("Try manually:")
                logger.warning(f"  sudo systemctl start {self.config.discrete_npu_service}")
                logger.warning(f"  sudo systemctl status {self.config.discrete_npu_service}")

            logger.debug("=" * 60)
            return False, []

        # Check if connector API is accessible
        logger.debug("")
        api_ready = self.check_connector_api()

        if not api_ready and auto_start_connector:
            logger.debug("Connector API not accessible, attempting to start...")
            if self.start_connector():
                # Wait a bit for API to be ready
                import time

                time.sleep(2)
                api_ready = self.check_connector_api()

        if not api_ready:
            logger.info("Note: Connector API is not ready but will be started when ARA model is selected")
        # Get available LLMs (filesystem + API)
        logger.debug("")
        llms = self.get_available_llms()  # Always call — filesystem works without API

        # Get device information only if API is ready
        if api_ready:
            self.get_devices_info()

        logger.debug("")
        if llms:
            logger.info("=" * 60)
            logger.info(f"✓ ARA DNPU AVAILABLE with {len(llms)} model(s):")
            for llm in llms:
                logger.info(f"  - {llm}")
            if not api_ready:
                logger.info("  Note: Connector API not ready yet — models listed from filesystem.")
                logger.info("        Connector will be started when you select an ARA model.")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning("ARA DNPU available but no models found")
            logger.warning(f"  Checked filesystem: {self.llm_base_path}")
            logger.warning(f"  Checked config: {self.connector_path}/server_config.json")
            logger.warning("=" * 60)

        return len(llms) > 0, llms

    def wait_for_connector_ready(self, timeout: int = 600, check_interval: int = 5) -> bool:
        """
        Wait for the connector API to become ready AND have models loaded.
        The connector is considered ready only when /v1/models returns at least one model.

        Args:
            timeout: Maximum time to wait in seconds (default: 10 minutes)
            check_interval: How often to check in seconds (default: 5 seconds)

        Returns:
            True if connector is ready with models, False if timeout
        """
        import time

        logger.info("=" * 60)
        logger.info("Waiting for ARA connector to be ready...")
        logger.info("This can take up to 10 minutes on first run")
        logger.info("=" * 60)

        start_time = time.time()
        elapsed = 0
        attempt = 0

        while elapsed < timeout:
            attempt += 1
            elapsed = time.time() - start_time

            try:
                url = f"{self.base_url}/v1/models"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])

                    if models:
                        # At least one model is listed = connector fully ready
                        logger.info("")
                        logger.info("=" * 60)
                        logger.info(f"✓ Connector API ready after {elapsed:.1f} seconds")
                        logger.info(f"  Models available: {[m.get('id') for m in models]}")
                        self.get_devices_info()
                        logger.info("=" * 60)
                        return True
                    else:
                        # API responds but no models yet — still loading
                        logger.debug(f"  API up but no models listed yet ({elapsed:.0f}s)")
                else:
                    logger.debug(f"  API not ready yet (status {response.status_code})")

            except Exception as e:
                logger.debug(f"  API not reachable yet: {type(e).__name__}")

            # Show progress every 30 seconds
            if attempt % 6 == 0:
                logger.info(f"  Still waiting for API + models... ({elapsed:.0f}s / {timeout}s)")

            time.sleep(check_interval)

        logger.error("")
        logger.error("=" * 60)
        logger.error(f"✗ Connector API did not become ready within {timeout} seconds")
        logger.error("=" * 60)
        return False

    def ensure_connector_ready(self, timeout: int = 600, model_name: str = None) -> bool:
        """
        Ensure connector is running, API is ready, and optionally wait for model.
        Start connector if needed and wait for it to be ready.

        Args:
            timeout: Maximum time to wait for connector (default: 10 minutes)
            model_name: Optional model name to wait for (without -ara suffix)

        Returns:
            True if ready, False otherwise
        """
        # Check if API is already ready
        if self.check_connector_api():
            logger.info("ARA connector API is already ready")

            # If model specified, check if it's loaded and ready
            if model_name:
                is_loaded, is_ready = self.check_model_loaded(model_name)
                if is_loaded and is_ready:
                    logger.info(f"Model '{model_name}' is already loaded and ready")
                    return True
                elif is_loaded and not is_ready:
                    logger.info(f"Model '{model_name}' is loaded but not ready, waiting...")
                    return self.wait_for_model_ready(model_name, timeout=timeout)
                else:
                    logger.info(f"Model '{model_name}' will be loaded on first request")
                    return True

            return True

        # Check if connector process is running
        if not self.check_connector_running():
            logger.info("Starting ARA connector...")
            if not self.start_connector():
                logger.error("Failed to start connector process")
                return False

        # Wait for API to be ready
        if not self.wait_for_connector_ready(timeout=timeout):
            return False

        # If model specified, wait for it to be ready
        if model_name:
            return self.wait_for_model_ready(model_name, timeout=timeout)

        return True
