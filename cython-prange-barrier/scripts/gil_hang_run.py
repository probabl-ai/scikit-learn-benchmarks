"""Call each function of the gil_hang extension in DIR in a subprocess, with a timeout to detect hangs."""
import subprocess, sys

d, label = sys.argv[1:3]
for f in ("target_read_after", "target_unused"):
    code = f"import sys; sys.path.insert(0, {d!r}); import gil_hang; print(gil_hang.{f}())"
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=20)
        print(label, f, "rc", r.returncode, r.stdout.strip(), r.stderr.strip()[-100:])
    except subprocess.TimeoutExpired:
        print(label, f, "HANG (killed after 20 s)")
