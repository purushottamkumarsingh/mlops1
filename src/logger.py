import logging
import os
from datetime import datetime

# Log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Log directory
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("✅ Logging has started successfully!")
    print(f"Logs are being saved to: {LOG_FILE_PATH}")
