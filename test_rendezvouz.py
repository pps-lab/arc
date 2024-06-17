
import utils.python_utils.rendezvouz as rendezvouz
import sys

hostname = 'localhost'
num_clients = 3

# get client id from args main
if len(sys.argv) > 1:
    client_id = int(sys.argv[1])

rendezvouz.sync(hostname, num_clients, client_id)

print("Done and continue!")