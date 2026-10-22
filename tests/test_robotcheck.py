import http.server
import socketserver
import threading

from puppy_mind import robotcheck


def _serve(body: bytes):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_real_web_video_server_is_recognised_and_a_printer_is_not():
    robot = _serve(b'<a href="/stream_viewer?topic=/usb_cam/image_raw">cam</a>')
    printer = _serve(b"<html>HP LaserJet admin</html>")
    try:
        assert robotcheck.is_web_video_server("127.0.0.1", robot.server_address[1])
        assert not robotcheck.is_web_video_server("127.0.0.1", printer.server_address[1])
    finally:
        robot.shutdown()
        printer.shutdown()


def test_find_robot_picks_rosbridge_hosts_and_web_video_servers_only(monkeypatch):
    open_ports = {("10.0.0.5", 9090), ("10.0.0.7", 8080), ("10.0.0.9", 8080)}
    monkeypatch.setattr(robotcheck, "_open", lambda ip, port, t=1.5: (ip, port) in open_ports)
    monkeypatch.setattr(robotcheck, "is_web_video_server", lambda ip, *a, **k: ip == "10.0.0.7")
    assert robotcheck.find_robot("10.0.0") == ["10.0.0.5", "10.0.0.7"]  # .9 (e.g. a printer) is excluded


def test_pi_like_hosts_matches_raspberry_pi_oui(monkeypatch):
    import subprocess
    arp = "  192.168.1.20   b8-27-eb-aa-bb-cc  dynamic\n  192.168.1.21   00-16-78-61-44-14  dynamic\n  192.168.2.9  dc-a6-32-00-11-22  dynamic\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {"stdout": arp})())
    assert robotcheck.pi_like_hosts("192.168.1") == [("192.168.1.20", "b8-27-eb-aa-bb-cc")]
