# A copy of the ODent5 add-on's engine client (addon/core/engine.py), for the smoke
# test: the protocol as the add-on speaks it. Keep it equal to the add-on's (a protocol change
# goes into both; the add-on's copy is the one in use). Its executable() finds the engine in
# bin/<platform>/ beside this folder, as the add-on finds it in its own bin/.
"""ODent — the client of odent_engine, the C++ program that does the CT and mesh
work out of Blender's process (no register).

The engine is started on first use and talks JSON lines on its stdin/stdout.
Voxels, slices and meshes stay in shared memory blocks the engine names in its
replies; `array()` maps one as a numpy array without copying it.

Only the Python standard library and numpy: nothing to install.
"""

import json
import os
import subprocess
import sys
import threading
from multiprocessing import shared_memory

import numpy as np  # type: ignore

API = 1
_EXE = "odent_engine.exe" if sys.platform == "win32" else "odent_engine"
_PLATFORM = "win-x64" if sys.platform == "win32" else "linux-x64"


class EngineError(RuntimeError):
    """The engine refused a command; the message says why."""


class Refused(Exception):
    """The data cannot be used; the message says why (shown to the doctor)."""


def doctor_message(er):
    """What to tell the doctor when the engine could not do a job (`er`, the
    exception): the cards' lines. The details go to the log."""
    if not os.path.isfile(executable()):
        return ["The ODent engine is not installed for this computer",
                f"({_PLATFORM}): install the full ODent package"]
    return ["The ODent engine could not do it:", str(er)[:120],
            "Open Log for the details"]


def executable():
    """The bundled engine: <addon>/bin/<platform>/odent_engine[.exe]."""
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(addon_dir, "bin", _PLATFORM, _EXE)


def ready_executable():
    """The engine to run as a program of its own (segment-teeth, register), its
    executable bit given back when a zip install dropped it; '' when it is missing."""
    exe = executable()
    if not os.path.isfile(exe):
        return ""
    if os.name != "nt" and not os.access(exe, os.X_OK):
        try:
            os.chmod(exe, os.stat(exe).st_mode | 0o755)
        except OSError:
            return ""
    return exe


class Engine:
    def __init__(self, exe=None):
        self.exe = exe or executable()
        self._proc = None
        self._lock = threading.Lock()
        self._caller = None   # the thread in a call (_send)
        self._next_id = 0
        self._blocks = {}   # shm name -> SharedMemory kept open while mapped
        self._cts = {}      # CT file -> its volume in the engine (load_ct)
        self.info = {}

    def start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not os.path.isfile(self.exe):
            raise EngineError(f"the ODent engine is missing: {self.exe}")
        if os.name != "nt" and not os.access(self.exe, os.X_OK):
            # Blender installs an addon zip with zipfile, which drops the
            # executable bit: give it back (the user owns the installed file)
            os.chmod(self.exe, os.stat(self.exe).st_mode | 0o755)
        self._proc = subprocess.Popen(
            [self.exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.info = self._send("ping", {})
        if self.info.get("api") != API:
            self.stop()
            raise EngineError(f"the ODent engine speaks version {self.info.get('api')}, "
                              f"the addon {API}: reinstall the addon")

    def stop(self):
        for shm in self._blocks.values():
            shm.close()
        self._blocks.clear()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send("quit", {})
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def call(self, cmd, on_progress=None, **params):
        """Run a command and return its result (a dict); EngineError if refused.
        A crashed engine is started again once. `on_progress(fraction)`: the
        command's progress lines (0..1) as they come, for the commands that send
        them (mesh.threshold); called from the thread that calls."""
        for attempt in (0, 1):
            self.start()
            try:
                return self._send(cmd, params, on_progress)
            except BrokenPipeError:
                if attempt:
                    raise EngineError("the ODent engine stopped")
                self._proc = None
                self._blocks.clear()
                self._cts.clear()
                if cmd != "image.load" and "volume" in params:
                    raise EngineError("the ODent engine stopped and lost its data")

    def _send(self, cmd, params, on_progress=None):
        # a call from inside a call on the same thread (an on_progress that called the
        # engine) would wait for itself on the lock: refused instead
        if self._caller == threading.get_ident():
            raise EngineError(f"the engine is busy with another command ({cmd} refused)")
        with self._lock:
            self._caller = threading.get_ident()
            try:
                return self._exchange(cmd, params, on_progress)
            finally:
                self._caller = None

    def _exchange(self, cmd, params, on_progress):
        """One request and its reply (the caller holds the lock)."""
        self._next_id += 1
        request = {"id": self._next_id, "cmd": cmd, "params": params}
        if on_progress is not None:
            request["progress"] = True            # lines {"id", "progress"} before the reply
        self._proc.stdin.write(json.dumps(request).encode("utf-8") + b"\n")
        self._proc.stdin.flush()
        while True:
            answer = self._proc.stdout.readline()
            if not answer:
                raise BrokenPipeError("no answer")
            reply = json.loads(answer.decode("utf-8"))
            if "progress" in reply and "ok" not in reply and "error" not in reply:
                if on_progress is not None:
                    on_progress(reply["progress"])
                continue
            break
        if "error" in reply:
            raise EngineError(reply["error"])
        return reply["ok"]

    def array(self, info):
        """The voxels of a volume (an engine reply) as a numpy array (z, y, x),
        in the engine's memory: valid until the volume is freed."""
        name = info["shm"]
        shm = self._blocks.get(name)
        if shm is None:
            shm = shared_memory.SharedMemory(name=name, track=False)
            self._blocks[name] = shm
        return np.ndarray(tuple(info["shape"]), dtype=np.dtype(info["dtype"]),
                          buffer=shm.buf[: info["nbytes"]])

    def load_ct(self, path):
        """The volume of a CT file in the engine, read once (until the file
        changes or the engine restarts)."""
        key = (os.path.abspath(path), os.path.getmtime(path))
        info = self._cts.get(key[0])
        if info is None or info[0] != key[1]:
            if info is not None:
                self.free(info[1])
            info = self._cts[key[0]] = (key[1], self.call("image.load", file=key[0]))
        return info[1]

    def adopt_ct(self, path, info):
        """Keep an engine volume as the CT of a file (load_ct gives it back)."""
        path = os.path.abspath(path)
        old = self._cts.pop(path, None)
        if old is not None and old[1]["volume"] != info["volume"]:
            self.free(old[1])
        self._cts[path] = (os.path.getmtime(path), info)

    def free_cts(self, keep=None):
        """Free the CT files' volumes the engine keeps (load_ct), but `keep`'s:
        the CT that is not shown any more (one CT per project)."""
        keep = os.path.abspath(keep) if keep else None
        for path in [p for p in self._cts if p != keep]:
            _mtime, info = self._cts.pop(path)
            self.free(info)

    def free(self, info):
        """Unmap and free a volume (an engine reply)."""
        self.release(info)
        try:
            self.call("image.free", volume=info["volume"])
        except EngineError:
            pass

    def release(self, info):
        """Unmap a volume's block on this side (call before freeing it)."""
        shm = self._blocks.pop(info["shm"], None)
        if shm is not None:
            shm.close()


_ENGINE = None


def engine():
    """The addon's engine (started on first call)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE
