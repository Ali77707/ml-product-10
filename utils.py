import logging
import re
from datetime import datetime

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)

def log_action(user_id, action, result=""):
    logging.info(f"user_id={user_id} | action={action} | result={result}")

def mask_phone(phone):
    if len(phone) >= 10:
        return phone[:5] + "***" + phone[-4:]
    return phone

def mask_name(name):
    parts = name.split()
    masked = []
    for part in parts:
        if len(part) <= 2:
            masked.append(part[0] + "*" * (len(part) - 1))
        else:
            masked.append(part[0] + "*" * (len(part) - 2) + part[-1])
    return " ".join(masked)

def get_logs(n=50):
    try:
        with open('app.log', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines[-n:]
    except FileNotFoundError:
        return ["Лог-файл не найден"]
