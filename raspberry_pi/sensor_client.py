import requests
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sensor_client.log'),
        logging.StreamHandler()
    ]
)

# Server configuration
SERVER_URL = 'http://your-server-ip:5000/api/push_sensor_data'  # Replace with your server's IP
RETRY_INTERVAL = 5  # Seconds to wait before retrying on failure
SEND_INTERVAL = 1   # Seconds between successful sends

# Import your sensor libraries here
# Example:
# import Adafruit_DHT  # For DHT temperature/humidity sensor
# import board
# import busio
# import adafruit_ads1x15.ads1115 as ADS
# from adafruit_ads1x15.analog_in import AnalogIn

def read_ph_sensor():
    """
    Read pH sensor value
    Replace this with your actual pH sensor reading code
    """
    try:
        # Example code for analog pH sensor:
        # i2c = busio.I2C(board.SCL, board.SDA)
        # ads = ADS.ADS1115(i2c)
        # ph_channel = AnalogIn(ads, ADS.P0)
        # voltage = ph_channel.voltage
        # ph = 7 - ((voltage - 2.5) / 0.18)  # Convert voltage to pH
        # return round(ph, 1)
        
        # For testing, return a dummy value
        return 6.5
    except Exception as e:
        logging.error(f"Error reading pH sensor: {e}")
        return None

def read_tds_sensor():
    """
    Read TDS sensor value
    Replace this with your actual TDS sensor reading code
    """
    try:
        # Example code for analog TDS sensor:
        # i2c = busio.I2C(board.SCL, board.SDA)
        # ads = ADS.ADS1115(i2c)
        # tds_channel = AnalogIn(ads, ADS.P1)
        # voltage = tds_channel.voltage
        # tds = voltage * 1000  # Convert voltage to ppm
        # return round(tds)
        
        # For testing, return a dummy value
        return 1000
    except Exception as e:
        logging.error(f"Error reading TDS sensor: {e}")
        return None

def read_turbidity_sensor():
    """
    Read turbidity sensor value
    Replace this with your actual turbidity sensor reading code
    """
    try:
        # Example code for analog turbidity sensor:
        # i2c = busio.I2C(board.SCL, board.SDA)
        # ads = ADS.ADS1115(i2c)
        # turbidity_channel = AnalogIn(ads, ADS.P2)
        # voltage = turbidity_channel.voltage
        # turbidity = voltage * 100  # Convert voltage to NTU
        # return round(turbidity)
        
        # For testing, return a dummy value
        return 5.0
    except Exception as e:
        logging.error(f"Error reading turbidity sensor: {e}")
        return None

def send_sensor_data():
    """
    Read all sensors and send data to server
    """
    try:
        # Read sensor values
        ph = read_ph_sensor()
        tds = read_tds_sensor()
        turbidity = read_turbidity_sensor()

        # Prepare data
        data = {
            'ph': ph,
            'tds': tds,
            'turbidity': turbidity,
            'timestamp': datetime.now().isoformat()
        }

        # Send data to server
        response = requests.post(SERVER_URL, json=data)
        response.raise_for_status()  # Raise exception for bad status codes
        
        logging.info(f"Data sent successfully: {data}")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending data to server: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False

def main():
    """
    Main loop to continuously read and send sensor data
    """
    logging.info("Starting sensor client...")
    
    while True:
        try:
            if send_sensor_data():
                time.sleep(SEND_INTERVAL)
            else:
                logging.warning(f"Failed to send data, retrying in {RETRY_INTERVAL} seconds...")
                time.sleep(RETRY_INTERVAL)
        except KeyboardInterrupt:
            logging.info("Sensor client stopped by user")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            time.sleep(RETRY_INTERVAL)

if __name__ == "__main__":
    main() 