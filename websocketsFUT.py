import websockets
import asyncio
import json
import time
import sys

class WebSocketClient():

    def __init__(self):
        pass

    async def connect(self):
        '''
            Connecting to webSocket server

            websockets.client.connect returns a WebSocketClientProtocol, which is used to send and receive messages
        '''
        self.connection = await websockets.client.connect('wss://fstream.binance.com/stream?streams=btcusdt_perpetual@continuousKline_15m')
        if self.connection.open:
            print('Connection stablished. Client correcly connected.')
            # Send greeting
            #await self.sendMessage('Hey server, this is webSocket client')
            return self.connection


    async def sendMessage(self, message):
        '''
            Sending message to webSocket server
        '''
        await self.connection.send(message)

    async def receiveMessage(self, connection):
        '''
            Receiving all server messages and handling them
        '''
        while True:
            try:
                message = await connection.recv()
                message_json=json.loads(message)
                print(json.dumps(message_json))
            except websockets.exceptions.ConnectionClosed or KeyboardInterrupt:
                print('Connection with server closed.')
                outfile.close()
                sys.exit()
                break

    async def heartbeat(self, connection):
        '''
        Sending heartbeat to server every 5 seconds
        Ping - pong messages to verify connection is alive
        '''
        while True:
            try:
                #time.sleep(250)
                await connection.send('pong')
                await asyncio.sleep(250)
            except websockets.exceptions.ConnectionClosed:
                print('Connection with server closed.')
                break

