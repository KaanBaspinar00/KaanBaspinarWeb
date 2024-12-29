import asyncio
import json
import logging
from datetime import datetime
from pydantic import BaseModel, ValidationError
from typing import List
import random
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define Pydantic model for acceleration data
class AccelerationData(BaseModel):
    x: float
    y: float
    z: float
    timestamp: float

# Variables to control data acquisition
acquisition_started = False
sensor_data: List[dict] = []
data_lock = asyncio.Lock()  # Async lock for thread safety

# Connection Manager for visualization clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List = []

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Client connected.")

    async def disconnect(self, websocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Client disconnected.")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                await self.disconnect(connection)

# Real-time WebSocket-based data retrieval
async def get_real_time_data(websocket):
    """Retrieve real-time data from the phone via WebSocket."""
    global sensor_data, acquisition_started

    try:
        async for message in websocket.iter_text():
            if acquisition_started:
                try:
                    # Parse the incoming data
                    data = AccelerationData.parse_raw(message)
                    async with data_lock:
                        sensor_data.append(data.dict())  # Append to the list
                    logger.info(f"Real-time data received: {data.dict()}")
                    yield data.dict()  # Yield the parsed data
                except ValidationError as e:
                    logger.error(f"Validation error: {e}")
            else:
                logger.info("Acquisition not started, skipping data.")
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")


# Sensor-related functions
async def start_acquisition():
    global acquisition_started
    async with data_lock:
        acquisition_started = True
    logger.info("Data acquisition started.")

async def stop_acquisition():
    global acquisition_started
    async with data_lock:
        acquisition_started = False
    logger.info("Data acquisition stopped.")

async def save_sensor_data():
    global sensor_data
    filename = f"sensor_data_{int(datetime.utcnow().timestamp())}.json"
    try:
        async with data_lock:
            with open(filename, 'w') as f:
                json.dump(sensor_data, f)
            sensor_data = []  # Clear data after saving
        logger.info(f"Data saved to {filename}.")
        return filename
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise Exception("An error occurred while saving data.")
