
import socket
from threading import Thread

RENDEZVOUZ_PORT=47683

def client_thread(conn, addr, all_connections, expected_clients):
    print(f"Connected by {addr}")
    conn.recv(1024)  # Wait for a ready signal from the client
    all_connections.append(conn)
    while len(all_connections) < expected_clients:
        pass  # Wait until all clients are connected
    conn.sendall(b"GO")  # Signal the client to proceed
    conn.close()

def rendezvous_server(port, expected_clients):
    all_connections = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', port))
        s.listen()
        print(f"Server listening on port {port}")
        threads = []
        while len(all_connections) < expected_clients:
            conn, addr = s.accept()
            thread = Thread(target=client_thread, args=(conn, addr, all_connections, expected_clients))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join() # ensure all clients have been notified
        s.close()


def notify_rendezvous_server(server_host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_host, port))
        print("Sending ready signal")
        s.sendall(b"READY")
        print("Waiting for go-ahead")
        data = s.recv(1024)
        if data == b"GO":
            print("Received go-ahead. Proceeding with the script.")
            return True
        return False

def sync(server_host, num_clients: int, client_id: int):
    if client_id == 0:
        print("Starting rendezvous server...")
        rendezvous_server(RENDEZVOUZ_PORT, num_clients)
    else:
        print("Connecting to rendezvous server...")
        if not notify_rendezvous_server(server_host, RENDEZVOUZ_PORT):
            print("Failed to sync with server!")
    print("Done with sync")
