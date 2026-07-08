#!/usr/bin/env python3
"""Proprioception — the Yahboom CMP10A IMU as Sprout's inner ear.

Turns accel/gyro/angle into a self-motion state (still / moving / rotating) + orientation,
enabling reafference: telling *the world moved* from *I moved*. Vendored lean reader
(11-byte 0x55-headed packets, 9600 baud on /dev/ttyUSB0) — packet format from
ai-dna-discovery/imu/yahboom_cmp10a.py. Fails open (no IMU → vision still works).
"""
from __future__ import annotations
import struct, threading, time, math

PORT = "/dev/ttyUSB0"
BAUD = 9600          # the CMP10A's 921600 upshift fails on this unit; 9600 is reliable
GYRO_ROT_DPS = 15.0  # gyro magnitude above this = rotating/turning
ACCEL_DEV_G = 0.15   # |accel|-1g above this = linear motion (beyond gravity)


def _parse(pkt: bytes):
    if len(pkt) != 11 or pkt[0] != 0x55 or (sum(pkt[:10]) & 0xFF) != pkt[10]:
        return None
    h, d = pkt[1], pkt[2:10]
    if h == 0x51:   # accel, ±16g
        return "accel", [struct.unpack('<h', d[i:i+2])[0] / 32768.0 * 16 for i in (0, 2, 4)]
    if h == 0x52:   # gyro, ±2000°/s
        return "gyro", [struct.unpack('<h', d[i:i+2])[0] / 32768.0 * 2000 for i in (0, 2, 4)]
    if h == 0x53:   # angle, ±180°
        return "angle", [struct.unpack('<h', d[i:i+2])[0] / 32768.0 * 180 for i in (0, 2, 4)]
    return None


class Proprioception(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.accel = [0.0, 0.0, 1.0]
        self.gyro = [0.0, 0.0, 0.0]
        self.angle = [0.0, 0.0, 0.0]
        self.ok = False
        self.err = None
        self.running = True
        self._buf = b""
        try:
            import serial
            self.ser = serial.Serial(PORT, BAUD, timeout=0.1)
            self.ok = True
        except Exception as e:
            self.err = str(e)

    def run(self):
        if not self.ok:
            return
        while self.running:
            try:
                n = self.ser.in_waiting
                if n:
                    self._buf += self.ser.read(n)
                while len(self._buf) >= 11:
                    s = self._buf.find(b"\x55")
                    if s < 0:
                        self._buf = b""; break
                    if s > 0:
                        self._buf = self._buf[s:]
                    if len(self._buf) < 11:
                        break
                    r = _parse(self._buf[:11])
                    if r:
                        setattr(self, r[0], r[1]); self._buf = self._buf[11:]
                    else:
                        self._buf = self._buf[1:]
            except Exception:
                pass
            time.sleep(0.02)

    def state(self) -> dict:
        amag = math.sqrt(sum(a*a for a in self.accel))
        gmag = math.sqrt(sum(g*g for g in self.gyro))
        if not self.ok:
            sm = "unknown"
        elif gmag > GYRO_ROT_DPS:
            sm = "rotating"
        elif abs(amag - 1.0) > ACCEL_DEV_G:
            sm = "moving"
        else:
            sm = "still"
        r, p, y = self.angle
        return {"self_motion": sm, "roll": round(r, 1), "pitch": round(p, 1), "yaw": round(y, 1),
                "accel_mag": round(amag, 3), "gyro_mag": round(gmag, 1), "ok": self.ok}

    def stop(self):
        self.running = False
        time.sleep(0.05)
        try:
            self.ser.close()
        except Exception:
            pass
