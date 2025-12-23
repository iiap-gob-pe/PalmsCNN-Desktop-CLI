# Archivo: aplicativo/app/single_instance.py
import socket
import sys

class SingleInstance:
    def __init__(self, port=55555):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = port
        self.is_running = False

    def check(self):
        """
        Intenta enlazar un socket al puerto local. 
        Si falla, significa que ya hay una instancia corriendo.
        """
        try:
            # Intentar enlazar al puerto en localhost
            self.socket.bind(('127.0.0.1', self.port))
            self.is_running = False
            return True
        except socket.error:
            # El puerto ya está en uso, hay otra instancia
            self.is_running = True
            return False

instance_checker = SingleInstance()