# Intel L515 LiDAR – Python Support (Unofficial)

This repository provides the required environment configuration, dependency pinning, and test scripts to enable **unofficial Python support** for the Intel RealSense **L515 LiDAR**, which is no longer supported by Intel’s recent Python SDK releases.

---

## Background

As of **2024**, Intel has officially discontinued support for the **L515 LiDAR**, and recent versions of `pyrealsense2` have dropped firmware compatibility for this device.

As a result:

- Installing the latest RealSense SDK or Python bindings causes the L515 to fail during stream initialization
- The device may appear in `rs-enumerate-devices`, but depth/RGB streams do not start
- OpenCV, ROS, and Python-based pipelines may fail silently or throw runtime errors

This repository addresses these issues by:

- Pinning known-compatible SDK, firmware, and Python versions
- Providing a working Conda environment (`env.yml`)
- Including test scripts to validate LiDAR streaming in Python

---

## What This Repository Provides

- Correct version mapping between:
  - `librealsense`
  - `pyrealsense2`
  - L515 firmware
- A predefined Conda environment to avoid dependency conflicts
- Python test scripts for:
  - Depth streaming
  - RGB streaming
  - OpenCV visualization
- A reproducible setup that works reliably on Linux (recommended)

---

## Environment Setup

**Do NOT install `pyrealsense2` using pip**, as it will install an incompatible version.

Always use the provided Conda environment.

### 1. Create the Environment

```bash
conda env create -f env.yml
conda activate l515/intel80/what-ever-it-is
```

The env.yml file pins compatible versions of:

- Python
- NumPy
- OpenCV
- pyrealsense2

# 2. Verify Device Connection

Before running any Python scripts, ensure the device is detected:

`rs-enumerate-devices`


You should see Intel RealSense L515 listed along with firmware details.

# Running the Test Scripts
Depth Stream Test
```bash
python test_depth.py
```

Displays the LiDAR depth stream using OpenCV.

RGB + Depth Stream Test
```bash
python test_rgb_depth.py
```

Runs synchronized RGB and depth streaming.

# Common Issues & Fixes
### Device Detected but No Stream

Ensure you are using the exact versions defined in env.yml

Do not upgrade pyrealsense2

A firmware downgrade may be required (refer to Intel archive firmware)

### ImportError: pyrealsense2
pip uninstall pyrealsense2


Then recreate the Conda environment from scratch using env.yml.

# Supported Platforms
- OS	Status
- Linux (Ubuntu 18.04 / 20.04)	Fully Tested
- Windows	Limited
- macOS	Not Supported [Did not test]

# Disclaimer

This repository provides unofficial support for the Intel L515 LiDAR.

Intel does not maintain or endorse this setup

Future OS, driver, or SDK updates may break compatibility

Use this setup only if continued Python-based access to the L515 is required



# Final Notes

If you upgrade any of the following:

- Python
- Conda
- RealSense SDK

Expect this setup to break.

For stability, strictly adhere to the provided environment configuration.
