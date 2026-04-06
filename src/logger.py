"""
src/logger.py
Cấu hình logging cho toàn bộ ứng dụng (7.2.5)
"""
import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("smartdoc.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("smartdoc")


logger = setup_logger()
