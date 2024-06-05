
import socket
from threading import Thread
import time

RENDEZVOUZ_PORT=47683

def client_thread(conn, addr, all_connections, expected_clients):
    print(f"Connected by {addr}", flush=True)
    conn.recv(1024)  # Wait for a ready signal from the client
    all_connections.append(conn)
    while len(all_connections) < expected_clients:
        time.sleep(1)
        pass  # Wait until all clients are connected
    conn.sendall(b"GO")  # Signal the client to proceed
    conn.close()

def rendezvous_server(port, expected_clients):
    all_connections = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', port))
        s.listen()
        s.settimeout(1)
        print(f"Server listening on port {port}", flush=True)
        threads = []
        while len(all_connections) < expected_clients:
            try:
                conn, addr = s.accept()
                thread = Thread(target=client_thread, args=(conn, addr, all_connections, expected_clients))
                thread.start()
                threads.append(thread)
            except socket.timeout:
                if len(all_connections) >= expected_clients:
                    break
        for thread in threads:
            thread.join() # ensure all clients have been notified
        print("All clients have connected. Shutting down server.", flush=True)
        s.close()


def notify_rendezvous_server(server_host, port, retries=1000, delay=5):
    attempts = 0
    while attempts < retries:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((server_host, port))
                s.sendall(b"READY")
                print("Waiting for go-ahead")
                data = s.recv(1024)
                if data == b"GO":
                    print("Received go-ahead. Proceeding with the script.", flush=True)
                    return True
                return False
        except socket.error as e:
            print(f"Connection attempt {attempts + 1} failed: {e}")
            time.sleep(delay)  # Wait before retrying
            attempts += 1
    return False

def sync(server_host, num_clients: int, client_id: int):
    if client_id == 0:
        print("Starting rendezvous server...", flush=True)
        rendezvous_server(RENDEZVOUZ_PORT, num_clients - 1) # -1 because the server is also ready
    else:
        print("Connecting to rendezvous server...", flush=True)
        if not notify_rendezvous_server(server_host, RENDEZVOUZ_PORT):
            print("Failed to sync with server!")
    print("Done with sync")
