# Setting Up Camera Passthrough to WSL2

## Prerequisites on Windows Side

### Step 1: Install usbipd-win (if not already installed)

Open an **Administrator PowerShell** on Windows and run:

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

Or download from: https://github.com/dorssel/usbipd-win/releases

### Step 2: List USB Devices

In **Administrator PowerShell**, run:

```powershell
usbipd list
```

Look for your camera device. It will show something like:
```
BUSID  VID:PID    DEVICE                                      STATE
2-3    046d:085b  Logitech BRIO                              Not shared
3-4    13d3:5666  USB Camera                                 Not shared
```

### Step 3: Share and Attach Camera to WSL2

1. First, share the camera (use the BUSID from the list):
```powershell
usbipd bind --busid 2-3
```

2. Then attach it to WSL2:
```powershell
usbipd attach --wsl --busid 2-3
```

### Step 4: Verify in WSL2

Back in WSL2, run:
```bash
ls -la /dev/video*
v4l2-ctl --list-devices
```

You should now see `/dev/video0` or similar.

### Step 5: Test Camera

```bash
# Quick test with v4l2
v4l2-ctl --device=/dev/video0 --all

# Or test with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera ready!' if cap.isOpened() else 'Camera not found'); cap.release()"
```

### To Detach Camera (when done)

In Windows PowerShell:
```powershell
usbipd detach --busid 2-3
```

## Troubleshooting

1. **"Access denied"** - Make sure PowerShell is running as Administrator
2. **"Device not found"** - Camera might be in use by Windows app, close all camera apps
3. **"WSL 2 is not running"** - Start WSL2 first
4. **Camera not showing in /dev/** - Try `sudo modprobe uvcvideo` in WSL2

## Note
The camera will need to be reattached after each WSL2 restart.