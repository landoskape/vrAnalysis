import socket
from argparse import ArgumentParser
from .app import create_app


def main():
    parser = ArgumentParser()
    parser.add_argument("--fast", default=False, action="store_true")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", default=5001, type=int, help="Port to run the server on")
    args = parser.parse_args()

    app = create_app(fast_mode=args.fast)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Interactive plot server running on http://{local_ip}:{args.port}")

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
