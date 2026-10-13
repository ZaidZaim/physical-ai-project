import http.server
import io
import socketserver
import threading
import time

from PIL import Image

from puppy_mind.sources import MjpegSource


def _jpeg(shade):
    b = io.BytesIO()
    Image.new("L", (64, 48), shade).save(b, "JPEG")
    return b.getvalue()


def test_mjpeg_source_keeps_only_newest_and_does_not_batch():
    frames = [_jpeg(10 * i) for i in range(1, 30)]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=f")
            self.end_headers()
            try:
                for j in frames:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.04)
            except OSError:
                pass

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = S(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    src = MjpegSource(f"http://127.0.0.1:{srv.server_address[1]}/stream")
    try:
        got, end = 0, time.time() + 1.0
        while time.time() < end:
            if src.read():
                got += 1
            time.sleep(0.005)
        assert got >= 15, got  # ~25 sent in 1 s; batching bug gave ~5
        assert src.read() in (None, src._latest)
    finally:
        src.close()
        srv.shutdown()
