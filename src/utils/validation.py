"""
Order parameter validation, input sanitization for CSV/API data, and detailed error reporting.
"""
import re

class OrderValidationError(Exception):
    pass

def validate_price(price):
    # Allow None for market orders
    if price is None:
        return
    if not isinstance(price, (int, float)) or price <= 0:
        raise OrderValidationError(f"Invalid price: {price}. Price must be a positive number.")


def validate_quantity(quantity):
    if not isinstance(quantity, int) or quantity <= 0:
        raise OrderValidationError(f"Invalid quantity: {quantity}. Quantity must be a positive integer.")


def validate_symbol(symbol):
    if not isinstance(symbol, str) or not re.match(r'^[A-Z][A-Z0-9]{0,12}$', symbol):
        raise OrderValidationError(f"Invalid symbol: {symbol}. Must be an uppercase string starting with letter (1-13 chars).")


def sanitize_csv_row(row):
    # Basic CSV input sanitization (expects iterable row)
    return [str(cell).strip().replace(',', '') for cell in row]


def sanitize_api_payload(payload):
    # Deep sanitize dictionary-based API response
    if isinstance(payload, dict):
        return {k: sanitize_api_payload(v) for k, v in payload.items()}
    elif isinstance(payload, list):
        return [sanitize_api_payload(i) for i in payload]
    elif isinstance(payload, str):
        return payload.strip()
    else:
        return payload

